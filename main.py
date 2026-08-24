import sys

if sys.platform == "win32":
    # LLM output can contain Unicode punctuation (curly quotes, em dashes,
    # narrow no-break spaces) that crashes rich's legacy Windows console
    # renderer under the default cp1252 codepage. Switch to UTF-8 before
    # anything prints.
    import os as _os
    _os.system("chcp 65001 >nul")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import copy
import gymnasium as gym
import numpy as np
import pandas as pd
import os
# base setting
from scenario.envScenario import EnvScenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.vectorStore import DrivingMemory
from LLMDriver.reflectionAgent import ReflectionAgent
from LLMDriver.reflection_choose_agent import Reflection_Choose_Agent
from LLMDriver.masterAgent import MasterAgent
from LLMDriver.verification import VerificationModule
from gymnasium.wrappers import RecordVideo
from langchain.callbacks import get_openai_callback
from loadConfig import load_openai_config, get_framework_flags

cfg = load_openai_config()
flags = get_framework_flags(cfg)

USE_MEMORY = flags["USE_MEMORY"]
REFLECTION = flags["REFLECTION"]
ENABLE_MASTER = flags["ENABLE_MASTER"]
ENABLE_VERIFICATION = flags["ENABLE_VERIFICATION"]
# Proposed dual-path stubs (manuscript Future Work; not evaluated)
ENABLE_DUAL_PATH = flags["ENABLE_DUAL_PATH"]
T_MAX_SECONDS = flags["T_MAX_SECONDS"]
ASYNC_REFLECTION = flags["ASYNC_REFLECTION"]

encode_type = 'sce_language'
db_path = os.environ.get("DB_PATH", "db/test")
result_folder = os.environ.get("RESULT_FOLDER", "./result/test")
few_shot_num = flags["FEW_SHOT_NUM"]
k_candidates = flags["K_CANDIDATES"]
simulation_duration = int(os.environ.get("SIMULATION_DURATION", "20"))
max_steps_per_episode = int(os.environ.get("MAX_STEPS_PER_EPISODE", str(simulation_duration)))

# environment setting
config={
    'KoMA-merge-generalization':
        {
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "TimeToCollision",

                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "target_speeds":np.linspace(0,40,9),
                },
            },
            "simulation_frequency": 100,  # [Hz]
            "policy_frequency": 2,  # [Hz] The number of actions performed per second
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 1400,  # [px]
            "screen_height": 200,  # [px]
            "centering_position": [0.5, 0.5],
            "scaling": 5.5,
            "show_trajectories": True,
            "render_agent": False,
            "offscreen_rendering": False,
            "other_vehicles_count": 5,
            "controlled_vehicles_count": 2
        }
}

agentMemory = None
verifier = None
if USE_MEMORY:
    agentMemory = DrivingMemory(encode_type=encode_type, db_path=db_path)
    if ENABLE_VERIFICATION:
        verifier = VerificationModule(
            tau_verify=flags["TAU_VERIFY"],
            lambda_time=flags["LAMBDA_TIME"],
            factual_use_llm=flags["FACTUAL_USE_LLM"],
            empty_fallback=flags["VERIFICATION_EMPTY_FALLBACK"],
            scenario_type=flags["SCENARIO_TYPE"],
            verbose=True,
        )

master = None
if ENABLE_MASTER:
    master = MasterAgent(
        delta_t_safe=flags["DELTA_T_SAFE"],
        lambda_coop=flags["LAMBDA_COOP"],
        lambda_conflict=flags["LAMBDA_CONFLICT"],
        conflict_only_llm=flags["MASTER_CONFLICT_ONLY_LLM"],
        use_llm=flags["MASTER_USE_LLM"],
        verbose=True,
    )

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
with open(result_folder + "/" + 'log.txt', 'a') as f:
    f.write(
        "result_folder {} USE_MEMORY={} REFLECTION={} ENABLE_MASTER={} "
        "ENABLE_VERIFICATION={} DUAL_PATH={} T_MAX={} ASYNC_REFLECTION={}\n".format(
            result_folder, USE_MEMORY, REFLECTION, ENABLE_MASTER,
            ENABLE_VERIFICATION, ENABLE_DUAL_PATH, T_MAX_SECONDS, ASYNC_REFLECTION
        )
    )
controlled_vehicle_number = config['KoMA-merge-generalization']["controlled_vehicles_count"]
episode = 0

while episode < simulation_duration:
    envType = 'KoMA-merge-generalization'
    env = gym.make(envType, render_mode='rgb_array')
    env.unwrapped.configure(config[envType])
    result_prefix = f"exp_{episode}"
    env = RecordVideo(env, result_folder, name_prefix=result_prefix)
    env.unwrapped.set_record_video_wrapper(env)
    obs, info = env.reset()
    env.render()
    sce = EnvScenario(env, envType)
    DA_list=[]
    DA1 = DriverAgent(sce, verbose=True)
    DA2 = DriverAgent(sce, verbose=True)
    DA_list.append(DA1)
    DA_list.append(DA2)

    if REFLECTION:
        RA = ReflectionAgent(verbose=True)
        RCA = Reflection_Choose_Agent(verbose=True)

    docs_list = [[] for i in range(controlled_vehicle_number)]
    efficiency_score_list = [[] for i in range(controlled_vehicle_number)]
    safety_score_list = [[] for i in range(controlled_vehicle_number)]
    collision_list = [0 for i in range(controlled_vehicle_number)]
    reward_list = []

    break_flag = False
    try:
        with get_openai_callback() as cb:
            already_decision_steps = 0
            previous_plan_list = [None, None]
            for j in range(0, max_steps_per_episode):
                collision_frame = -1
                obs = np.array(obs, dtype=float)

                # --- Master coordination (critical path when enabled) ---
                directives = {}
                if ENABLE_MASTER and master is not None:
                    ego_states, idm_states = MasterAgent.extract_states_from_env(env.unwrapped, sce)
                    proposed_goals = {
                        str(i): "Navigate safely, cooperate with other ego agents, avoid collisions"
                        for i in range(controlled_vehicle_number)
                    }
                    # Dual-path stub: when ENABLE_DUAL_PATH and T_MAX>0, Master is conflict-only
                    # (already the default via MASTER_CONFLICT_ONLY_LLM). No wall-clock abort yet.
                    directives = master.coordinate(
                        ego_states=ego_states,
                        proposed_goals=proposed_goals,
                        idm_states=idm_states,
                        enable=True,
                    )

                action_list=[]
                for i in range(controlled_vehicle_number):
                    docs = docs_list[i]
                    sce_descrip = sce.describe(i)
                    avail_action = sce.availableActionsDescription(i)
                    previous_plan = previous_plan_list[i]
                    print(sce_descrip)
                    fewshot_messages = []
                    fewshot_answers = []
                    fewshot_actions = []
                    if USE_MEMORY and agentMemory is not None:
                        retrieve_k = k_candidates if ENABLE_VERIFICATION else few_shot_num
                        if ENABLE_VERIFICATION and verifier is not None:
                            scored = agentMemory.retrieveMemoryWithScores(
                                sce, i, top_k=retrieve_k, query_text=sce_descrip)
                            candidates = [(meta, dist) for meta, dist, _ in scored]
                            fewshot_results = verifier.filter_memories(
                                candidates=candidates,
                                context=sce_descrip,
                                current_time=float(j),
                                top_k=few_shot_num,
                                apply_filter=True,
                            )
                        else:
                            fewshot_results = agentMemory.retriveMemory(
                                sce, i, few_shot_num)
                        for fewshot_result in fewshot_results:
                            fewshot_messages.append(
                                fewshot_result["human_question"])
                            fewshot_answers.append(fewshot_result["LLM_response"])
                            fewshot_actions.append(fewshot_result["action"])
                            mode_action = max(
                                set(fewshot_actions), key=fewshot_actions.count)
                            mode_action_count = fewshot_actions.count(mode_action)

                    master_prompt = None
                    if ENABLE_MASTER and str(i) in directives:
                        master_prompt = directives[str(i)].to_prompt_block()

                    action, response, human_question, fewshot_answer, new_plan = DA_list[i].few_shot_decision(
                        scenario_description=sce_descrip, available_actions=avail_action,
                        fewshot_messages=fewshot_messages,
                        driving_attentions="Drive safely and avoid collisions.No matter how fast the two vehicles are, changing lanes onto the lane of a parallel vehicle will always result in a collision",
                        fewshot_answers=fewshot_answers,
                        previous_plan=previous_plan_list[i],
                        master_directive=master_prompt)
                    action_list.append(action)
                    previous_plan_list[i] = new_plan
                    docs.append({
                        "controlled_vehicle_id": i,
                        "simulation_time": j,
                        "sce_description": sce_descrip,
                        "human_question": human_question,
                        "response": response,
                        "plan": new_plan,
                        "action": action,
                        "sce": copy.deepcopy(sce),
                        "master_directive": master_prompt or "",
                    })
                action = tuple(action_list)
                obs, reward, done, info, _ = env.step(action)
                reward_list.append(float(np.mean(reward)) if hasattr(reward, "__iter__") else float(reward))
                speed, efficiency_score, safety_score = sce.evaluation(controlled_vehicle_number)
                for k in range(controlled_vehicle_number):
                    docs_list[k][-1]["efficiency_score"] = efficiency_score[k]
                    docs_list[k][-1]["safety_score"] = safety_score[k]
                    docs_list[k][-1]["speed"] = speed[k]
                    efficiency_score_list[k].append(efficiency_score[k])
                    safety_score_list[k].append(safety_score[k])
                already_decision_steps += 1
                env.render()
                env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()
                if done:
                    print("[red]Simulation done with running steps: [/red] ", j)
                    collision_frame = j
                    for i in range(controlled_vehicle_number):
                        collision_list[i] = env.controlled_vehicles[i].crashed
                    print(collision_list)
                    break
    finally:
        print("==========Simulation {} Done==========".format(episode))
        print(cb)
        print("Simulation done")

        for controlled_vehicle in range(controlled_vehicle_number):
            is_collision = collision_list[controlled_vehicle]
            docs = docs_list[controlled_vehicle]
            e_list = efficiency_score_list[controlled_vehicle]
            s_list = safety_score_list[controlled_vehicle]
            path = result_folder + "/" + f"Simulation_{episode}.csv".format(episode=episode)
            pd.DataFrame(docs).to_csv(path_or_buf=path, mode='a')
            # Fix: only write memory when USE_MEMORY constructed a store
            if REFLECTION and USE_MEMORY and agentMemory is not None:
                if is_collision:  # has collision
                    i = collision_frame
                    corrected_response = RA.reflection(
                        docs[i]["human_question"], docs[i]["response"], e_list[i], s_list[i], is_collision)
                    agentMemory.addMemory(
                        docs[i]["sce_description"],
                        docs[i]["human_question"],
                        corrected_response,
                        docs[i]["plan"],
                        docs[i]["action"],
                        docs[i]["sce"],
                        comments="collision-mistake-correction"
                        )
                else:
                    try:
                        wrong_action = int(RCA.reflection_choose(e_list, s_list))
                    except ValueError:
                        print("[yellow]reflection_choose output not parseable as int; assuming no mistake.[/yellow]")
                        wrong_action = -1
                    if wrong_action == -1:
                        for i in range(0, len(docs)):
                            agentMemory.addMemory(
                                docs[i]["sce_description"],
                                docs[i]["human_question"],
                                docs[i]["response"],
                                docs[i]["plan"],
                                docs[i]["action"],
                                docs[i]["sce"],
                                comments="no-mistake-direct"
                            )
                    else:
                        corrected_response = RA.reflection(
                            docs[wrong_action]["human_question"], docs[wrong_action]["response"], e_list[wrong_action], s_list[wrong_action], False)
                        agentMemory.addMemory(
                            docs[wrong_action]["sce_description"],
                            docs[wrong_action]["human_question"],
                            corrected_response,
                            docs[wrong_action]["plan"],
                            docs[wrong_action]["action"],
                            docs[wrong_action]["sce"],
                            comments="mistake-correction"
                        )
            elif REFLECTION and not USE_MEMORY:
                print("[yellow]REFLECTION=true but USE_MEMORY=false; skipping memory writes.[/yellow]")

        # Episode-level metrics for ablation aggregation (reward, factual consistency, collision rate)
        episode_factual_consistency = (
            verifier.mean_factual_score() if (ENABLE_VERIFICATION and verifier is not None) else float("nan")
        )
        if ENABLE_VERIFICATION and verifier is not None:
            verifier.reset_episode_stats()
        summary_path = result_folder + "/episode_summary.csv"
        summary_row = pd.DataFrame([{
            "episode": episode,
            "steps": already_decision_steps,
            "total_reward": float(np.sum(reward_list)) if reward_list else 0.0,
            "mean_reward": float(np.mean(reward_list)) if reward_list else 0.0,
            "collision": int(any(collision_list)),
            "factual_consistency": episode_factual_consistency,
        }])
        summary_row.to_csv(
            summary_path, mode='a', index=False, header=not os.path.exists(summary_path)
        )

        episode += 1
        env.close()
