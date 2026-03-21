"""
agent/agent_loop.py
====================
Orchestrates the four tools at the end of each semester.

Decision logic:
  1. New comments?        → refresh_sentiment
  2. New students?        → cold_start_handler per new student
  3. New OEs?             → cold_start_handler per new OE
  4. Always               → retrain_model
  5. After retrain        → shap_eval

The agent remembers past actions via agent_memory.csv and
adjusts decisions based on what worked previously.

Usage:
  from agent.agent_loop import AgentLoop

  agent = AgentLoop()
  agent.run(semester=6)
"""

import os
import json
import pandas as pd
from datetime import datetime

from agent.tools import (retrain_model, refresh_sentiment,
                          shap_eval, cold_start_handler)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

RESULTS_DIR   = "results"
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
MEMORY_PATH   = f"{RESULTS_DIR}/agent_memory.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────

class AgentLoop:
    """
    Runs at the end of each semester.
    Decides which tools to call based on:
      - What changed this semester
      - What worked in previous semesters (memory)
    """

    def __init__(self):
        self.memory = self._load_memory()

    # ─────────────────────────────────────────────────────────
    # MEMORY
    # ─────────────────────────────────────────────────────────

    def _load_memory(self) -> list:
        """Load past semester actions and outcomes."""
        if os.path.exists(MEMORY_PATH):
            df = pd.read_csv(MEMORY_PATH)
            return df.to_dict(orient="records")
        return []

    def _save_memory(self, record: dict):
        """Append one semester's summary to memory."""
        df  = pd.DataFrame([record])
        if os.path.exists(MEMORY_PATH):
            df.to_csv(MEMORY_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(MEMORY_PATH, mode="w", header=True, index=False)

    def _get_last_ndcg(self) -> float:
        """Return NDCG@10 from the most recent semester in memory."""
        if not self.memory:
            return 0.0
        return float(self.memory[-1].get("ndcg_after", 0.0))

    # ─────────────────────────────────────────────────────────
    # DETECTION HELPERS
    # ─────────────────────────────────────────────────────────

    def _detect_new_comments(self, semester: int) -> bool:
        """
        Check if new comments exist for current semester
        that haven't been processed yet.
        """
        comments_path = f"{RAW_DIR}/course_comments.csv"
        scored_path   = f"{PROCESSED_DIR}/scored_comments.csv"

        if not os.path.exists(scored_path):
            return True

        comments_df = pd.read_csv(comments_path)
        scored_df   = pd.read_csv(scored_path)

        # New comments = more rows in raw than in scored
        return len(comments_df) > len(scored_df)

    def _detect_new_students(self, semester: int) -> list:
        """
        Return list of student_ids who have no OE history yet
        and are expected to pick an OE this semester.
        """
        students_df    = pd.read_csv(f"{RAW_DIR}/students.csv")
        interaction_df = pd.read_csv(f"{PROCESSED_DIR}/interaction_matrix.csv")

        students_with_history = set(
            interaction_df[interaction_df["is_negative"] == 0]["student_id"].unique()
        )

        # Students in the system but with no OE history
        all_students = set(students_df["student_id"].tolist())
        new_students = list(all_students - students_with_history)

        return new_students

    def _detect_new_oes(self, semester: int) -> list:
        """
        Return list of oe_ids marked is_new_oe=1 for current semester.
        """
        oe_info_df = pd.read_csv(f"{RAW_DIR}/oe_info.csv")
        new_oes    = oe_info_df[
            (oe_info_df["is_new_oe"]            == 1) &
            (oe_info_df["available_semester"]   == semester)
        ]["oe_id"].tolist()

        return new_oes

    # ─────────────────────────────────────────────────────────
    # REASON — decide what needs to be done
    # ─────────────────────────────────────────────────────────

    def reason(self, semester: int) -> dict:
        """
        Inspect current state and return a plan of actions.
        """
        print(f"\n{'='*50}")
        print(f"  Agent reasoning — end of semester {semester}")
        print(f"{'='*50}")

        plan = {
            "semester"          : semester,
            "refresh_sentiment" : False,
            "new_students"      : [],
            "new_oes"           : [],
            "retrain"           : True,    # always retrain at end of semester
            "shap"              : True,    # always evaluate features after retrain
        }

        # Check for new comments
        if self._detect_new_comments(semester):
            plan["refresh_sentiment"] = True
            print(f"  Detected new comments → will refresh sentiment")
        else:
            print(f"  No new comments detected")

        # Check for new students
        new_students = self._detect_new_students(semester)
        if new_students:
            plan["new_students"] = new_students
            print(f"  Detected {len(new_students)} new students → cold start")
        else:
            print(f"  No new students detected")

        # Check for new OEs
        new_oes = self._detect_new_oes(semester)
        if new_oes:
            plan["new_oes"] = new_oes
            print(f"  Detected {len(new_oes)} new OEs → cold start")
        else:
            print(f"  No new OEs detected")

        print(f"\n  Plan: {json.dumps({k: v for k, v in plan.items() if k != 'new_students'}, indent=2)}")
        return plan

    # ─────────────────────────────────────────────────────────
    # ACT — execute the plan
    # ─────────────────────────────────────────────────────────

    def act(self, plan: dict) -> dict:
        """
        Execute the plan by calling tools in the correct order.
        Order matters:
          1. Sentiment first (prof_features must be fresh before retrain)
          2. Cold start (register new entities before retrain)
          3. Retrain (uses latest features + interactions)
          4. SHAP (evaluates the freshly trained model)
        """
        semester = plan["semester"]
        outcomes = {"semester": semester, "actions": []}

        # Step 1 — Refresh sentiment if needed
        if plan["refresh_sentiment"]:
            print(f"\n--- Step 1: Refresh Sentiment ---")
            result = refresh_sentiment(semester)
            outcomes["actions"].append({
                "tool"  : "refresh_sentiment",
                "result": result,
            })
        else:
            print(f"\n--- Step 1: Refresh Sentiment — skipped ---")

        # Step 2 — Cold start for new students
        if plan["new_students"]:
            print(f"\n--- Step 2: Cold Start — {len(plan['new_students'])} new students ---")
            for student_id in plan["new_students"][:5]:   # cap at 5 for logging
                result = cold_start_handler(student_id=student_id,
                                            semester=semester)
                outcomes["actions"].append({
                    "tool"  : f"cold_start_student_{student_id}",
                    "result": result,
                })
        else:
            print(f"\n--- Step 2: Cold Start Students — skipped ---")

        # Step 3 — Cold start for new OEs
        if plan["new_oes"]:
            print(f"\n--- Step 3: Cold Start — {len(plan['new_oes'])} new OEs ---")
            for oe_id in plan["new_oes"]:
                result = cold_start_handler(oe_id=oe_id, semester=semester)
                outcomes["actions"].append({
                    "tool"  : f"cold_start_oe_{oe_id}",
                    "result": result,
                })
        else:
            print(f"\n--- Step 3: Cold Start OEs — skipped ---")

        # Step 4 — Retrain (always)
        print(f"\n--- Step 4: Retrain Model ---")
        retrain_result = retrain_model(semester)
        outcomes["actions"].append({
            "tool"  : "retrain_model",
            "result": retrain_result,
        })
        outcomes["ndcg_after"]  = retrain_result["new_ndcg"]
        outcomes["ndcg_before"] = retrain_result["old_ndcg"]   # pass directly, avoid memory pollution
        outcomes["replaced"]    = retrain_result["replaced"]

        # Step 5 — SHAP evaluation
        print(f"\n--- Step 5: SHAP Evaluation ---")
        shap_result = shap_eval(semester)
        outcomes["actions"].append({
            "tool"  : "shap_eval",
            "result": shap_result,
        })
        outcomes["sentiment_rank"] = shap_result.get("sentiment_rank", -1)

        return outcomes

    # ─────────────────────────────────────────────────────────
    # OBSERVE — check if things improved
    # ─────────────────────────────────────────────────────────

    def observe(self, outcomes: dict) -> dict:
        """
        Compare this semester's outcomes to previous semester.
        Uses actual checkpoint NDCG values — not memory entries
        which can be polluted from multiple test runs.
        """
        semester    = outcomes["semester"]
        ndcg_after  = outcomes.get("ndcg_after",  0.0)
        ndcg_before = outcomes.get("ndcg_before", self._get_last_ndcg())
        replaced    = outcomes.get("replaced", False)
        ndcg_delta  = round(ndcg_after - ndcg_before, 4)

        sentiment_rank = outcomes.get("sentiment_rank", -1)

        print(f"\n{'='*50}")
        print(f"  Agent observation — semester {semester}")
        print(f"{'='*50}")
        print(f"  NDCG@10 before   : {ndcg_before:.4f}")
        print(f"  NDCG@10 after    : {ndcg_after:.4f}")
        print(f"  Delta            : {ndcg_delta:+.4f}")
        print(f"  Model replaced   : {outcomes.get('replaced', False)}")

        if sentiment_rank > 0:
            total = outcomes["actions"][-1]["result"].get("total_features", -1)
            print(f"  Sentiment rank   : {sentiment_rank} / {total}")
            if sentiment_rank > total * 0.7:
                print(f"  Warning: sentiment score is low importance "
                      f"— consider reviewing comment quality")

        # Learning summary
        if ndcg_delta > 0:
            observation = f"Model improved by {ndcg_delta:+.4f} this semester"
        elif ndcg_delta == 0:
            observation = "Model performance unchanged"
        else:
            observation = (f"Model degraded by {ndcg_delta:.4f} — "
                           f"rolled back to previous checkpoint")

        print(f"\n  Observation: {observation}")

        memory_record = {
            "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "semester"      : semester,
            "ndcg_before"   : ndcg_before,
            "ndcg_after"    : ndcg_after,
            "ndcg_delta"    : ndcg_delta,
            "replaced"      : outcomes.get("replaced", False),
            "sentiment_rank": sentiment_rank,
            "observation"   : observation,
        }

        return memory_record

    # ─────────────────────────────────────────────────────────
    # RUN — full reason → act → observe cycle
    # ─────────────────────────────────────────────────────────

    def run(self, semester: int):
        """
        Full agent cycle for one semester.
        Called at the end of each semester when new data is available.
        """
        print(f"\n{'#'*50}")
        print(f"  Agent Loop — Semester {semester}")
        print(f"{'#'*50}")

        # Reason
        plan = self.reason(semester)

        # Act
        outcomes = self.act(plan)

        # Observe
        memory_record = self.observe(outcomes)

        # Remember
        self._save_memory(memory_record)
        self.memory.append(memory_record)

        print(f"\n✓ Agent cycle complete for semester {semester}")
        print(f"  Memory saved to {MEMORY_PATH}")

        return memory_record


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = AgentLoop()

    # Simulate end of semester 6
    # (in production this runs automatically when new data arrives)
    result = agent.run(semester=6)

    print(f"\n--- Agent Memory ---")
    if os.path.exists(MEMORY_PATH):
        memory_df = pd.read_csv(MEMORY_PATH)
        print(memory_df.to_string(index=False))
