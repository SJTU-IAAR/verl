def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # Extract question information if available
    question = ""
    if extra_info:
        question = extra_info.get("question", extra_info.get("query", ""))
    
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source == "taco":
        from . import taco_reward
        res = taco_reward.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    # Using qa_em module to handle all QA tasks
    elif data_source in ["nq", "nq_search", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]:
        from . import qa_em
        res = qa_em.compute_score_em(solution_str, ground_truth, return_dict=True)
    elif data_source == "xhpang_search":
        from . import xhpang_search_reward
        # Enhanced xhpang_search with xverify and OTC support
        res = xhpang_search_reward.compute_score(
            solution_str, 
            ground_truth, 
            return_dict=True,
            question=question,
            extra_info=extra_info,
            use_xverify=True,
            use_otc=False,
            otc_method="grpo"
        )
    elif data_source == "numina_math":
        from . import numina_math_reward
        # Enhanced numina_math with xverify and OTC support  
        res = numina_math_reward.compute_score(
            solution_str, 
            ground_truth, 
            return_dict=True,
            question=question,
            extra_info=extra_info,
            use_xverify=True,
            use_otc=False,
            otc_method="grpo"
        )
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


