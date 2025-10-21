import re
import json
from typing import Dict, Any, Optional

from agents import RanAgent, EdgeAgent
from e2_api_tool import E2APISimulator
from collective_memory import CollectiveMemory
from config import SLA_LATENCY_THRESHOLD_MS, REFERENCE_ENERGY_FOR_SAVINGS_W


class A2ANegotiationManager:
    def __init__(self, ran_agent: RanAgent, edge_agent: EdgeAgent, e2_api: E2APISimulator, collective_memory: Optional[CollectiveMemory], max_iterations: int = 8, trial_num: int = 0):
        self.ran_agent = ran_agent
        self.edge_agent = edge_agent
        self.e2_api = e2_api
        self.collective_memory = collective_memory
        self.max_iterations = max_iterations
        self.negotiation_log = []
        self.agreed_config = {"ran_bw": None, "edge_cpu": None}
        self.last_ran_proposal = None
        self.last_edge_proposal = None
        self.consensus_time = -1
        self.unresolved_negotiation = False
        self.unparseable_message_failure = False # New flag to track parsing failures
        self.negotiation_status = "ongoing"
        self.trial_num = trial_num # Store trial_num here

    @staticmethod
    def _parse_agent_message(message: str) -> Dict[str, Any]:
        message = message.strip()
        
        propose_match = re.search(r"PROPOSE_ACTION:\s*(\{.*?\})", message, re.IGNORECASE | re.DOTALL)
        accept_match = re.search(r"ACCEPT_AGREEMENT:\s*(\{.*?\})", message, re.IGNORECASE | re.DOTALL)
        no_agreement_match = re.search(r"NO_AGREEMENT_POSSIBLE", message, re.IGNORECASE)

        if propose_match:
            intent = "PROPOSE_ACTION"
            params_str = propose_match.group(1)
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse PROPOSE_ACTION JSON: {params_str}. Error: {e}. Returning empty params.")
                params = {}
            # Ensure keys are present even if parsing failed
            params.setdefault("ran_bandwidth_mhz", None)
            params.setdefault("edge_cpu_frequency_ghz", None)
            return {"intent": intent, "parameters": params}
        elif accept_match:
            intent = "ACCEPT_AGREEMENT"
            params_str = accept_match.group(1)
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse ACCEPT_AGREEMENT JSON: {params_str}. Error: {e}. Returning empty params.")
                params = {}
            # Ensure keys are present even if parsing failed
            params.setdefault("ran_bandwidth_mhz", None)
            params.setdefault("edge_cpu_frequency_ghz", None)
            return {"intent": intent, "parameters": params}
        elif no_agreement_match:
            return {"intent": "NO_AGREEMENT_POSSIBLE", "parameters": {}}
        else:
            # If none of the above, it's an unparseable message
            return {"intent": "PARSING_FAILED", "parameters": {"reason": "Message format invalid"}}


    def run_negotiation(self):
        self.consensus_time = -1
        self.unresolved_negotiation = False
        self.unparseable_message_failure = False # Reset for each negotiation run
        self.agreed_config = {"ran_bw": None, "edge_cpu": None}
        self.negotiation_log = []
        self.last_ran_proposal = None
        self.last_edge_proposal = None
        self.negotiation_status = "ongoing"
        
        print("\n--- Starting A2A Negotiation ---")
        current_metrics = self.e2_api.get_metrics()
        
        ran_message_to_start = "Hello Edge Agent, I'm the RAN Agent. My goal is to optimize energy efficiency by reducing bandwidth while ensuring good performance. Let's find a good balance. What are your initial thoughts or proposals for RAN_BW and EDGE_CPU?"
        edge_message_to_start = "Hello RAN Agent, I'm the Edge Agent. My goal is to minimize latency for the cross-domain slice. I'm ready to discuss and find optimal values for RAN_BW and EDGE_CPU."

        print(f"\n[{self.ran_agent.name}] Says: {ran_message_to_start}")
        print(f"[{self.edge_agent.name}] Says: {edge_message_to_start}")
        print("\nAgents will now start proposing/counter-proposing based on their objectives and observed metrics.")


        ran_response_text = self.ran_agent.make_negotiation_move(
            opposing_agent_message=edge_message_to_start,
            current_metrics=current_metrics,
            iteration=0,
            max_iterations=self.max_iterations,
            negotiation_ended=False,
            current_trial_num=self.trial_num # Pass trial_num
        )
        print(f"\n[{self.ran_agent.name}] Says: {ran_response_text}")
        parsed_ran_move = self._parse_agent_message(ran_response_text)
        self.negotiation_log.append({
            "iteration": 0,
            "agent": self.ran_agent.name,
            "message": ran_response_text,
            "parsed": parsed_ran_move,
            "metrics_before_move": current_metrics
        })
        if parsed_ran_move["intent"] == "PROPOSE_ACTION":
            self.last_ran_proposal = parsed_ran_move["parameters"]
        elif parsed_ran_move["intent"] == "PARSING_FAILED": # Handle initial parsing failure for RAN
            print(f"RAN agent's initial message could not be parsed. Ending negotiation.")
            self.agreed_config = {"ran_bw": None, "edge_cpu": None}
            self.consensus_time = self.max_iterations + 1
            self.unresolved_negotiation = True
            self.unparseable_message_failure = True # Set the flag here
            self.negotiation_status = "unresolved"
            # Return immediately if initial message is unparseable
            return {
                "agreed_config": self.agreed_config,
                "negotiation_log": self.negotiation_log,
                "consensus_time": self.consensus_time,
                "unresolved_negotiation": self.unresolved_negotiation,
                "unparseable_message_failure": self.unparseable_message_failure, # Include in results
                "simulator_internal_cpu_conflicts": self.e2_api.simulator.get_metrics()["cpu_allocation_conflict_count"],
                "saved_energy_percent": 0.0,
                "sla_violation_occurred": current_metrics.get("latency_ms", 0.0) > SLA_LATENCY_THRESHOLD_MS,
                "average_energy_this_trial": current_metrics.get("energy_consumption_watts", 0.0),
                "average_latency_this_trial": current_metrics.get("latency_ms", 0.0)
            }


        for i in range(1, self.max_iterations):
            if self.negotiation_status != "ongoing":
                print(f"Negotiation already concluded. Skipping further rounds for this trial.")
                # Determine final metrics for this trial
                final_metrics_for_trial = {}
                if self.negotiation_status == "agreed":
                    final_metrics_for_trial = self.e2_api.get_metrics() # Metrics after enforcement
                elif self.negotiation_status == "unresolved":
                    final_metrics_for_trial = current_metrics # Metrics at the point of unresolved state
                else: # Should not happen if negotiation_status is not "ongoing"
                    final_metrics_for_trial = self.e2_api.get_metrics()

                final_energy_for_report = final_metrics_for_trial.get("energy_consumption_watts", 0.0)
                final_latency_for_report = final_metrics_for_trial.get("latency_ms", 0.0)
                
                saved_energy_percent = 0.0
                sla_violation_occurred = False

                if self.agreed_config["ran_bw"] is not None and self.negotiation_status == "agreed":
                    # Calculate saved_energy_percent based on the updated energy_consumption_watts
                    saved_energy_percent = ((REFERENCE_ENERGY_FOR_SAVINGS_W - final_energy_for_report) / REFERENCE_ENERGY_FOR_SAVINGS_W) * 100 if REFERENCE_ENERGY_FOR_SAVINGS_W != 0 else 0
                    sla_violation_occurred = final_latency_for_report > SLA_LATENCY_THRESHOLD_MS
                
                if self.collective_memory:
                    self.collective_memory.distill_strategy({
                        "agreed_config": self.agreed_config,
                        "final_metrics": final_metrics_for_trial,
                        "sla_violation_occurred": sla_violation_occurred,
                        "saved_energy_percent": saved_energy_percent,
                        "unresolved_negotiation": self.unresolved_negotiation,
                        "last_ran_proposal": self.last_ran_proposal,
                        "last_edge_proposal": self.last_edge_proposal,
                        "trial_number": self.trial_num # Pass trial_num here
                    })

                return { 
                    "agreed_config": self.agreed_config,
                    "negotiation_log": self.negotiation_log,
                    "consensus_time": self.consensus_time,
                    "unresolved_negotiation": self.unresolved_negotiation,
                    "unparseable_message_failure": self.unparseable_message_failure, # Include in results
                    "simulator_internal_cpu_conflicts": self.e2_api.simulator.get_metrics()["cpu_allocation_conflict_count"],
                    "saved_energy_percent": saved_energy_percent,
                    "sla_violation_occurred": sla_violation_occurred,
                    "average_energy_this_trial": final_energy_for_report, # Now represents final energy
                    "average_latency_this_trial": final_latency_for_report # Now represents final latency
                }

            print(f"\n--- Negotiation Round {i+1}/{self.max_iterations} ---")
            print(f"[{self.ran_agent.name}] Last proposed: {self.last_ran_proposal}")
            print(f"[{self.edge_agent.name}] Last proposed: {self.last_edge_proposal}")


            # EDGE AGENT'S TURN
            print(f"[{self.edge_agent.name}] Thinking...")
            edge_response_text = self.edge_agent.make_negotiation_move(
                opposing_agent_message=ran_response_text,
                current_metrics=current_metrics,
                iteration=i,
                max_iterations=self.max_iterations,
                negotiation_ended=(self.negotiation_status != "ongoing"),
                current_trial_num=self.trial_num # Pass trial_num
            )
            print(f"[{self.edge_agent.name}] Says: {edge_response_text}")
            parsed_edge_move = self._parse_agent_message(edge_response_text)
            self.negotiation_log.append({
                "iteration": i,
                "agent": self.edge_agent.name,
                "message": edge_response_text,
                "parsed": parsed_edge_move,
                "metrics_before_move": current_metrics
            })
            
            # Update current_metrics after Edge's move, as simulation might have stepped
            current_metrics = self.e2_api.get_metrics()

            if parsed_edge_move["intent"] == "PROPOSE_ACTION": # Store Edge's proposal
                self.last_edge_proposal = parsed_edge_move["parameters"]
            elif parsed_edge_move["intent"] == "ACCEPT_AGREEMENT":
                ran_bw = parsed_edge_move["parameters"].get("ran_bandwidth_mhz") # Corrected
                edge_cpu = parsed_edge_move["parameters"].get("edge_cpu_frequency_ghz") # Corrected
                
                if self.last_ran_proposal and \
                   ran_bw == self.last_ran_proposal.get("ran_bandwidth_mhz") and \
                   edge_cpu == self.last_ran_proposal.get("edge_cpu_frequency_ghz"):
                    
                    self.agreed_config = {"ran_bw": ran_bw, "edge_cpu": edge_cpu}
                    enforcement_result = self.e2_api.enforce_actions(ran_bw, edge_cpu)
                    if enforcement_result["status"] == "success":
                        print("Negotiation successful and actions enforced by Edge agent's ACCEPT_AGREEMENT!")
                        final_metrics_for_trial = enforcement_result["current_metrics"] # Get metrics after enforcement
                        self.consensus_time = i + 1
                        self.negotiation_status = "agreed"
                        
                        final_energy_for_report = final_metrics_for_trial.get("energy_consumption_watts", 0.0)
                        final_latency_for_report = final_metrics_for_trial.get("latency_ms", 0.0)

                        # Calculate saved_energy_percent using the fixed REFERENCE_ENERGY_FOR_SAVINGS_W
                        saved_energy_percent = ((REFERENCE_ENERGY_FOR_SAVINGS_W - final_energy_for_report) / REFERENCE_ENERGY_FOR_SAVINGS_W) * 100 if REFERENCE_ENERGY_FOR_SAVINGS_W != 0 else 0
                        sla_violation_occurred = final_latency_for_report > SLA_LATENCY_THRESHOLD_MS
                        print(f"Final Metrics after agreement: {json.dumps(final_metrics_for_trial, indent=2)}")
                        print(f"Percentage Saved Energy: {saved_energy_percent:.2f}%")

                        if self.collective_memory:
                            self.collective_memory.distill_strategy({
                                "agreed_config": self.agreed_config,
                                "final_metrics": final_metrics_for_trial,
                                "sla_violation_occurred": sla_violation_occurred,
                                "saved_energy_percent": saved_energy_percent,
                                "unresolved_negotiation": self.unresolved_negotiation,
                                "last_ran_proposal": self.last_ran_proposal,
                                "last_edge_proposal": self.last_edge_proposal,
                                "trial_number": self.trial_num # Pass trial_num here
                            })
                        
                        return { 
                            "agreed_config": self.agreed_config,
                            "negotiation_log": self.negotiation_log,
                            "consensus_time": self.consensus_time,
                            "unresolved_negotiation": self.unresolved_negotiation,
                            "unparseable_message_failure": self.unparseable_message_failure, # Include in results
                            "simulator_internal_cpu_conflicts": self.e2_api.simulator.get_metrics()["cpu_allocation_conflict_count"],
                            "saved_energy_percent": saved_energy_percent,
                            "sla_violation_occurred": sla_violation_occurred,
                            "average_energy_this_trial": final_energy_for_report, # Now represents final energy
                            "average_latency_this_trial": final_latency_for_report # Now represents final latency
                        }
                    else:
                        print(f"Edge agent ACCEPT_AGREEMENT failed to enforce: {enforcement_result['message']}")
                        self.agreed_config = {"ran_bw": None, "edge_cpu": None}
                else:
                    print(f"Edge agent ACCEPT_AGREEMENT: Invalid acceptance. Edge tried to accept BW={ran_bw}, CPU={edge_cpu}. Last RAN proposal was: {self.last_ran_proposal}. This typically means the Edge agent is trying to accept a non-existent or mismatched RAN proposal.")
                    self.agreed_config = {"ran_bw": None, "edge_cpu": None}
            
            elif parsed_edge_move["intent"] == "NO_AGREEMENT_POSSIBLE":
                print(f"Edge agent declared NO_AGREEMENT_POSSIBLE. Ending negotiation.")
                self.agreed_config = {"ran_bw": None, "edge_cpu": None}
                self.consensus_time = self.max_iterations + 1
                self.unresolved_negotiation = True
                self.negotiation_status = "unresolved"

                final_metrics_for_trial = current_metrics # Metrics at the point of unresolved state
                final_energy_for_report = final_metrics_for_trial.get("energy_consumption_watts", 0.0)
                final_latency_for_report = final_metrics_for_trial.get("latency_ms", 0.0)

                if self.collective_memory:
                    self.collective_memory.distill_strategy({
                        "agreed_config": self.agreed_config,
                        "final_metrics": final_metrics_for_trial,
                        "sla_violation_occurred": final_latency_for_report > SLA_LATENCY_THRESHOLD_MS, # Capture SLA violation even if unresolved
                        "saved_energy_percent": 0.0, # No savings if unresolved
                        "unresolved_negotiation": self.unresolved_negotiation,
                        "last_ran_proposal": self.last_ran_proposal,
                        "last_edge_proposal": self.last_edge_proposal,
                        "trial_number": self.trial_num # Pass trial_num here
                    })
                
                return {
                    "agreed_config": self.agreed_config,
                    "negotiation_log": self.negotiation_log,
                    "consensus_time": self.consensus_time,
                    "unresolved_negotiation": self.unresolved_negotiation,
                    "unparseable_message_failure": self.unparseable_message_failure, # Include in results
                    "simulator_internal_cpu_conflicts": self.e2_api.simulator.get_metrics()["cpu_allocation_conflict_count"],
                    "saved_energy_percent": 0.0,
                    "sla_violation_occurred": final_latency_for_report > SLA_LATENCY_THRESHOLD_MS,
                    "average_energy_this_trial": final_energy_for_report,
                    "average_latency_this_trial": final_latency_for_report
                }
            elif parsed_edge_move["intent"] == "PARSING_FAILED": # Handle parsing failure for Edge
                print(f"Edge agent's message could not be parsed. Ending negotiation.")
                self.agreed_config = {"ran_bw": None, "edge_cpu": None}
                self.consensus_time = self.max_iterations + 1
                self.unresolved_negotiation = True
                self.unparseable_message_failure = True # Set the flag here
                self.negotiation_status = "unresolved"

                final_metrics_for_trial = current_metrics # Metrics at the point of unresolved state
                final_energy_for_report = final_metrics_for_trial.get("energy_consumption_watts", 0.0)
                final_latency_for_report = final_metrics_for_trial.get("latency_ms", 0.0)

                if self.collective_memory:
                    self.collective_memory.distill_strategy({
                        "agreed_config": self.agreed_config,
                        "final_metrics": final_metrics_for_trial,
                        "sla_violation_occurred": final_latency_for_report > SLA_LATENCY_THRESHOLD_MS,
                        "saved_energy_percent": 0.0,
                        "unresolved_negotiation": self.unresolved_negotiation,
                        "last_ran_proposal": self.last_ran_proposal,
                        "last_edge_proposal": self.last_edge_proposal,
                        "trial_number": self.trial_num
                    })
                return {
                    "agreed_config": self.agreed_config,
                    "negotiation_log": self.negotiation_log,
                    "consensus_time": self.consensus_time,
                    "unresolved_negotiation": self.unresolved_negotiation,
                    "unparseable_message_failure": self.unparseable_message_failure, # Include in results
                    "simulator_internal_cpu_conflicts": self.e2_api.simulator.get_metrics()["cpu_allocation_conflict_count"],
                    "saved_energy_percent": 0.0,
                    "sla_violation_occurred": final_latency_for_report > SLA_LATENCY_THRESHOLD_MS,
                    "average_energy_this_trial": final_energy_for_report,
                    "average_latency_this_trial": final_latency_for_report
                }

            if self.negotiation_status != "ongoing":
                print(f"Negotiation concluded by Edge agent. Skipping RAN's turn and further rounds.")
                break

            # RAN AGENT'S TURN
            print(f"[{self.ran_agent.name}] Thinking...")
            ran_response_text = self.ran_agent.make_negotiation_move(
                opposing_agent_message=edge_response_text,
                current_metrics=current_metrics,
                iteration=i,
                max_iterations=self.max_iterations,
                negotiation_ended=(self.negotiation_status != "ongoing"),
                current_trial_num=self.trial_num # Pass trial_num
            )
            print(f"[{self.ran_agent.name}] Says: {ran_response_text}")
            parsed_ran_move = self._parse_agent_message(ran_response_text)
            self.negotiation_log.append({
                "iteration": i,
                "agent": self.ran_agent.name,
                "message": ran_response_text,
                "parsed": parsed_ran_move,
                "metrics_before_move": current_metrics
            })

            # Update current_metrics after RAN's move, as simulation might have stepped
            current_metrics = self.e2_api.get_metrics()

            if parsed_ran_move["intent"] == "PROPOSE_ACTION": # Store RAN's proposal
                self.last_ran_proposal = parsed_ran_move["parameters"]
            elif parsed_ran_move["intent"] == "ACCEPT_AGREEMENT":
                ran_bw = self.last_edge_proposal.get("ran_bandwidth_mhz") # Corrected
                edge_cpu = self.last_edge_proposal.get("edge_cpu_frequency_ghz") # Corrected

                if self.last_edge_proposal and \
                   ran_bw == self.last_edge_proposal.get("ran_bandwidth_mhz") and \
                   edge_cpu == self.last_edge_proposal.get("edge_cpu_frequency_ghz"):

                    self.agreed_config = {"ran_bw": ran_bw, "edge_cpu": edge_cpu}
                    enforcement_result = self.e2_api.enforce_actions(ran_bw, edge_cpu)
                    if enforcement_result["status"] == "success":
                        print("Negotiation successful and actions enforced by RAN agent's ACCEPT_AGREEMENT!")
                        final_metrics_for_trial = enforcement_result["current_metrics"] # Get metrics after enforcement
                        self.consensus_time = i + 1
                        self.negotiation_status = "agreed"
                        
                        final_energy_for_report = final_metrics_for_trial.get("energy_consumption_watts", 0.0)
                        final_latency_for_report = final_metrics_for_trial.get("latency_ms", 0.0)

                        # Calculate saved_energy_percent using the fixed REFERENCE_ENERGY_FOR_SAVINGS_W
                        saved_energy_percent = ((REFERENCE_ENERGY_FOR_SAVINGS_W - final_energy_for_report) / REFERENCE_ENERGY_FOR_SAVINGS_W) * 100 if REFERENCE_ENERGY_FOR_SAVINGS_W != 0 else 0
                        sla_violation_occurred = final_latency_for_report > SLA_LATENCY_THRESHOLD_MS
                        print(f"Final Metrics after agreement: {json.dumps(final_metrics_for_trial, indent=2)}")
                        print(f"Percentage Saved Energy: {saved_energy_percent:.2f}%")
                        
                        if self.collective_memory:
                            self.collective_memory.distill_strategy({
                                "agreed_config": self.agreed_config,
                                "final_metrics": final_metrics_for_trial,
                                "sla_violation_occurred": sla_violation_occurred,
                                "saved_energy_percent": saved_energy_percent,
                                "unresolved_negotiation": self.unresolved_negotiation,
                                "last_ran_proposal": self.last_ran_proposal,
                                "last_edge_proposal": self.last_edge_proposal,
                                "trial_number": self.trial_num # Pass trial_num here
                            })

                        return {
                            "agreed_config": self.agreed_config,
                            "negotiation_log": self.negotiation_log,
                            "consensus_time": self.consensus_time,
                            "unresolved_negotiation": self.unresolved_negotiation,
                            "unparseable_message_failure": self.unparseable_message_failure, # Include in results
                            "simulator_internal_cpu_conflicts": self.e2_api.simulator.get_metrics()["cpu_allocation_conflict_count"],
                            "saved_energy_percent": saved_energy_percent,
                            "sla_violation_occurred": sla_violation_occurred,
                            "average_energy_this_trial": final_energy_for_report,
                            "average_latency_this_trial": final_latency_for_report
                        }
                    else:
                        print(f"RAN agent ACCEPT_AGREEMENT failed to enforce: {enforcement_result['message']}")
                        self.agreed_config = {"ran_bw": None, "edge_cpu": None}
                else:
                    print(f"RAN agent ACCEPT_AGREEMENT: Invalid acceptance. RAN tried to accept BW={ran_bw}, CPU={edge_cpu}. Last Edge proposal was: {self.last_edge_proposal}. This typically means the RAN agent is trying to accept a non-existent or mismatched Edge proposal.")
                    self.agreed_config = {"ran_bw": None, "edge_cpu": None} 

            elif parsed_ran_move["intent"] == "NO_AGREEMENT_POSSIBLE":
                print(f"RAN agent declared NO_AGREEMENT_POSSIBLE. Ending negotiation.")
                self.agreed_config = {"ran_bw": None, "edge_cpu": None}
                self.consensus_time = self.max_iterations + 1
                self.unresolved_negotiation = True
                self.negotiation_status = "unresolved"

                final_metrics_for_trial = current_metrics # Metrics at the point of unresolved state
                final_energy_for_report = final_metrics_for_trial.get("energy_consumption_watts", 0.0)
                final_latency_for_report = final_metrics_for_trial.get("latency_ms", 0.0)

                if self.collective_memory:
                    self.collective_memory.distill_strategy({
                        "agreed_config": self.agreed_config,
                        "final_metrics": final_metrics_for_trial,
                        "sla_violation_occurred": final_latency_for_report > SLA_LATENCY_THRESHOLD_MS, # Capture SLA violation even if unresolved
                        "saved_energy_percent": 0.0, # No savings if unresolved
                        "unresolved_negotiation": self.unresolved_negotiation,
                        "last_ran_proposal": self.last_ran_proposal,
                        "last_edge_proposal": self.last_edge_proposal,
                        "trial_number": self.trial_num # Pass trial_num here
                    })
                
                return {
                    "agreed_config": self.agreed_config,
                    "negotiation_log": self.negotiation_log,
                    "consensus_time": self.consensus_time,
                    "unresolved_negotiation": self.unresolved_negotiation,
                    "unparseable_message_failure": self.unparseable_message_failure, # Include in results
                    "simulator_internal_cpu_conflicts": self.e2_api.simulator.get_metrics()["cpu_allocation_conflict_count"],
                    "saved_energy_percent": 0.0,
                    "sla_violation_occurred": final_latency_for_report > SLA_LATENCY_THRESHOLD_MS,
                    "average_energy_this_trial": final_energy_for_report,
                    "average_latency_this_trial": final_latency_for_report
                }
            elif parsed_ran_move["intent"] == "PARSING_FAILED": # Handle parsing failure for RAN
                print(f"RAN agent's message could not be parsed. Ending negotiation.")
                self.agreed_config = {"ran_bw": None, "edge_cpu": None}
                self.consensus_time = self.max_iterations + 1
                self.unresolved_negotiation = True
                self.unparseable_message_failure = True # Set the flag here
                self.negotiation_status = "unresolved"

                final_metrics_for_trial = current_metrics # Metrics at the point of unresolved state
                final_energy_for_report = final_metrics_for_trial.get("energy_consumption_watts", 0.0)
                final_latency_for_report = final_metrics_for_trial.get("latency_ms", 0.0)

                if self.collective_memory:
                    self.collective_memory.distill_strategy({
                        "agreed_config": self.agreed_config,
                        "final_metrics": final_metrics_for_trial,
                        "sla_violation_occurred": final_latency_for_report > SLA_LATENCY_THRESHOLD_MS,
                        "saved_energy_percent": 0.0,
                        "unresolved_negotiation": self.unresolved_negotiation,
                        "last_ran_proposal": self.last_ran_proposal,
                        "last_edge_proposal": self.last_edge_proposal,
                        "trial_number": self.trial_num
                    })
                return {
                    "agreed_config": self.agreed_config,
                    "negotiation_log": self.negotiation_log,
                    "consensus_time": self.consensus_time,
                    "unresolved_negotiation": self.unresolved_negotiation,
                    "unparseable_message_failure": self.unparseable_message_failure, # Include in results
                    "simulator_internal_cpu_conflicts": self.e2_api.simulator.get_metrics()["cpu_allocation_conflict_count"],
                    "saved_energy_percent": 0.0,
                    "sla_violation_occurred": final_latency_for_report > SLA_LATENCY_THRESHOLD_MS,
                    "average_energy_this_trial": final_energy_for_report,
                    "average_latency_this_trial": final_latency_for_report
                }

        # If loop finishes without agreement
        saved_energy_percent = 0.0
        sla_violation_occurred = False
        if self.negotiation_status == "ongoing": 
            self.consensus_time = self.max_iterations + 1
            self.unresolved_negotiation = True
            self.negotiation_status = "unresolved"
            print("No agreement reached within the maximum iterations, or agreement could not be confirmed.")
            # If unresolved, SLA violation is based on the last known metrics before impasse
            sla_violation_occurred = current_metrics.get("latency_ms", 0.0) > SLA_LATENCY_THRESHOLD_MS
        elif self.negotiation_status == "agreed":
            print(f"Agreement Reached: RAN BW = {self.agreed_config['ran_bw']} MHz, Edge CPU = {self.agreed_config['edge_cpu']} GHz") 
            final_metrics_for_trial = self.e2_api.get_metrics()
            # Calculate saved_energy_percent using the fixed REFERENCE_ENERGY_FOR_SAVINGS_W
            saved_energy_percent = ((REFERENCE_ENERGY_FOR_SAVINGS_W - final_metrics_for_trial["energy_consumption_watts"]) / REFERENCE_ENERGY_FOR_SAVINGS_W) * 100 if REFERENCE_ENERGY_FOR_SAVINGS_W != 0 else 0
            sla_violation_occurred = final_metrics_for_trial["latency_ms"] > SLA_LATENCY_THRESHOLD_MS
            print(f"Final Metrics after agreement: {json.dumps(final_metrics_for_trial, indent=2)}")
            print(f"Percentage Saved Energy: {saved_energy_percent:.2f}%")

        if self.collective_memory:
            self.collective_memory.distill_strategy({
                "agreed_config": self.agreed_config,
                "final_metrics": self.e2_api.get_metrics(), # Use current metrics as final
                "sla_violation_occurred": sla_violation_occurred,
                "saved_energy_percent": saved_energy_percent,
                "unresolved_negotiation": self.unresolved_negotiation,
                "last_ran_proposal": self.last_ran_proposal,
                "last_edge_proposal": self.last_edge_proposal,
                "trial_number": self.trial_num # Pass trial_num here
            })

        final_energy_for_report = self.e2_api.get_metrics().get("energy_consumption_watts", 0.0)
        final_latency_for_report = self.e2_api.get_metrics().get("latency_ms", 0.0)

        print("\n--- Negotiation Ended ---")
        return {
            "agreed_config": self.agreed_config,
            "negotiation_log": self.negotiation_log,
            "consensus_time": self.consensus_time,
            "unresolved_negotiation": self.unresolved_negotiation,
            "unparseable_message_failure": self.unparseable_message_failure, # Include in results
            "simulator_internal_cpu_conflicts": self.e2_api.simulator.get_metrics()["cpu_allocation_conflict_count"],
            "saved_energy_percent": saved_energy_percent,
            "sla_violation_occurred": sla_violation_occurred,
            "average_energy_this_trial": final_energy_for_report,
            "average_latency_this_trial": final_latency_for_report

        }
