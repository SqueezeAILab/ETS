import argparse
import json
import re
import time
from sglang import function, gen, RuntimeEndpoint, system, user, assistant
import fcntl
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import math
import yaml
import torch
import torch.nn.functional as F
import threading
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
import threading
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import requests
from scipy.cluster.hierarchy import linkage, fcluster

# for tracking stats across threads
lock = threading.Lock()

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_prompts(args):
    test_cases = read_jsonl(args.input_path)
    if args.num_samples is not None:
        test_cases = test_cases[:args.num_samples]
    prompts = []
    for test in test_cases:
        prompts.append(test["problem"])
    return prompts, test_cases

# global variables for tracking statistics
global_num_model_calls = 0
global_num_tokens_generated = 0
global_kv_size = 0

class TreeNode:
    def __init__(self, id, state, score, num_step_tokens=0, parent=None):
        self.id = id
        self.state = state
        self.text_ = state.text()
        self.score_ = score
        self.parent = parent
        self.leaf_ = False
        self.cum_tokens = 0
        self.num_step_tokens = num_step_tokens
        if parent is not None:
            if "The answer is" in self.text_ or "The final answer is:" in self.text_ or "Therefore, the final answer is" in self.text_[len(parent.get_text()):]:
                self.leaf_ = True
        if parent is not None:
            self.depth = parent.get_depth() + 1
            self.cum_tokens = parent.get_cum_tokens() + num_step_tokens
        else:
            self.depth = 0
            self.cum_tokens = num_step_tokens

        # for tracking per-beam width
        self.dvts_id = None

    def get_id(self):
        return self.id

    def get_parent(self):
        return self.parent

    def get_text(self):
        return self.text_

    def get_state(self):
        return self.state

    def get_depth(self):
        return self.depth

    def get_score(self):
        return self.score_

    def is_leaf(self):
        return self.leaf_

    def get_cum_tokens(self):
        return self.cum_tokens

    def set_dvts_id(self, dvts_id):
        self.dvts_id = dvts_id

    def get_dvts_id(self):
        return self.dvts_id

class Tree:
    def __init__(self, root_state, paras, reward_backend, multimodel, config):
        # initialize tokenizer to compute question length
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        num_root_tokens = len(tokenizer(root_state.text()))

        self.size_ = 1
        self.nodes = []
        self.paras = paras
        self.reward_backend = reward_backend
        self.root_ = TreeNode(0,root_state, 1.0, num_step_tokens=num_root_tokens)
        self.remaining_width = paras["width"]
        self.init_width = paras["width"]
        self.history_list = []
        self.running_list = []
        self.depth_nodes = [[] for i in range(100)]
        self.nodes.append(self.root_)
        self.depth_nodes[0].append(self.root_)

        # beam search params
        self.fixedwidth = paras.get("fixedwidth", -1)

        # for DVTS
        if paras['select_method'] == "dvts":
            if self.fixedwidth > -1:
                num_subtrees = self.fixedwidth
                tree_width = self.remaining_width // num_subtrees
            else:
                num_subtrees = int(math.ceil(math.sqrt(self.remaining_width)))
                tree_width = self.remaining_width // num_subtrees
            self.remaining_width_dvts = [ tree_width for i in range(num_subtrees)]
        else:
            self.remaining_width_dvts = None

        # cost model params
        self.lambdac = paras.get("lambdac", 0)
        self.lambdas = paras.get("lambdas", 0)

        # embedding model
        self.model = multimodel

        # set question length
        self.question_length = len(self.root_.get_text())


    def reset_running_list(self):
        self.running_list = []

    def get_running_list(self):
        return self.running_list

    def get_history_list(self):
        return self.history_list

    def get_nodes(self):
        return self.nodes

    def expand(self, node, wid):
        state = node.get_state()
        forks = state.fork(wid)
        depth = node.get_depth()
        for fork in forks:
            fork.set_score_backend(self.reward_backend)
            if self.paras["policy_model_type"] == "llemma":
                fork += gen("step", self.paras["max_step_tokens"], stop="ки", temperature=self.paras["temperature"])
                fork += gen("score", max_tokens=0, forward_only=True, logits_require_id=8094)
            elif self.paras["policy_model_type"] == "mistral":
                fork += gen("step", self.paras["max_step_tokens"], stop="ки", temperature=self.paras["temperature"])
                fork += gen("score", max_tokens=0, forward_only=True, logits_require_id=[648,387,12902]) # plus_tag_id, minus_tag_id, step_tag_id
            elif self.paras["policy_model_type"] == "llama":
                fork += assistant(gen("step", self.paras["max_step_tokens"], stop="\n\n", temperature=self.paras["temperature"]))
                fork += gen("score", max_tokens=0, forward_only=True, logits_require_id=[10,12,-3]) # plus_tag_id, minus_tag_id, step_tag_id (-3 position)
            else:
                assert(False)

            self.running_list.append((fork, node))
            self.history_list.append(fork)

            with lock:
                global global_num_model_calls
                global_num_model_calls += 1

    def insert(self, state, parent):

        num_step_tokens = state.get_meta_info("step")["completion_tokens"]
        with lock:
            global global_num_tokens_generated
            global_num_tokens_generated += num_step_tokens

        if state.scores() == [] or state.scores == None:
            return

        if isinstance(state.scores(), list):
            score = state.scores()[-1]
        else:
            assert(isinstance(state.scores(), float))
            score = state.scores()

        new_node = TreeNode(self.size_, state, score, num_step_tokens, parent)
        self.size_ += 1
        depth = new_node.get_depth()
        self.depth_nodes[depth].append(new_node)
        self.nodes.append(new_node)
        return

    def get_kv_size(self, nodes):

        # node list for tracking memory operations
        leaf_node_lengths = []
        non_leaf_node_lengths = {}
        for i in range(len(nodes)):
            leaf_node = nodes[i]
            leaf_node_lengths.append(leaf_node.num_step_tokens)

            # use dict to prune out duplicates
            node = leaf_node.parent
            while node != None:
                non_leaf_node_lengths[node.id] = node.num_step_tokens
                node = node.parent

        # track kv cache size at each step
        kv_size = sum(leaf_node_lengths)
        for k,v in non_leaf_node_lengths.items():
            kv_size += v
        return kv_size

    def select_beam_search(self, node_list, node_weights, width, depth):
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()
        nodes = []
        nodes_kept = []
        select_num = []

        if self.fixedwidth > 0:
            keep = self.fixedwidth
        else:
            keep = int(math.ceil(math.sqrt(width)))

        # expand straight up to width for depth of 0
        if depth == 0:
            assert(len(node_list) == 1)
            num = width
        else:
            num = int(math.ceil(width / keep))

        idx = 0
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            if idx < keep:
                nodes_kept.append(pair[0])
                select_num.append(num)
            else:
                select_num.append(0)
            idx += 1

        # get KV cache size
        kv_size = self.get_kv_size(nodes)
        with lock:
            global global_kv_size
            global_kv_size += kv_size

        return nodes, select_num

    def select_dvts(self, node_list, node_weights, width, depth):
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]

        # use fixed number of subtrees
        if self.fixedwidth > 0:
            num_subtrees = self.fixedwidth
        else:
            num_subtrees = int(math.ceil(math.sqrt(self.paras["width"])))

        share_list = {}
        share_list_weight = {}

        # handle depth=1
        if depth == 1:
            # split into subtrees
            if len(node_weights) < num_subtrees:
                # duplicate up to subtrees
                import itertools
                cycled_nodes = itertools.cycle(node_list)
                cycled_weights = itertools.cycle(node_weights)

                node_list = list(itertools.islice(cycled_nodes, num_subtrees))
                node_weights = list(itertools.islice(cycled_weights, num_subtrees))
                node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]

            # split into subtrees
            split_size = len(node_weights) // num_subtrees
            num_remaining = len(node_weights) - (split_size * num_subtrees) # handle extras if number is not exactly divisible

            start_idx = 0
            for i in range(0,num_subtrees):
                end_idx = start_idx + split_size
                if num_remaining > 0:
                    end_idx += 1
                    num_remaining -= 1

                # loop over nodes in (start_idx, end_idx) and pick the one with highest weight
                pair = node_weight_pair_list[start_idx]
                share_list[i] = pair[0].id
                share_list_weight[i] = pair[1]
                for j in range(start_idx, end_idx):
                    pair = node_weight_pair_list[j]
                    pair[0].set_dvts_id(i) # extra - tracking remaining width per-subtree
                    if pair[1] > share_list_weight[i]:
                        share_list[i] = pair[0].id
                        share_list_weight[i] = pair[1]

                start_idx = end_idx
        else:
            # select nodes to keep (one per parent)
            for pair in node_weight_pair_list:
                node = pair[0]
                weight = pair[1]
                if node.parent is not None:
                    node.set_dvts_id(node.parent.get_dvts_id()) # extra - tracking remaining width per-subtree
                    if node.parent.id not in share_list:
                        share_list[node.parent.id] = node.id
                        share_list_weight[node.parent.id] = weight
                    else:
                        if weight > share_list_weight[node.parent.id]:
                            share_list[node.parent.id] = node.id
                            share_list_weight[node.parent.id] = weight
                else:
                    assert (depth == 0)

        # share_list / share_list_weight now has the trajectories to continue
        nodes = []
        nodes_kept = []
        select_num = []
        if depth == 0: # first iteration - expand up to full width
            num = self.paras["width"]
        else:
            num_list = self.remaining_width_dvts

        # get retained leaf node ids
        if depth > 0:
            keep_node_ids = []
            for k,v in share_list.items():
                keep_node_ids.append(v)

        # keep one node per parent and expand
        if depth == 0: # first iteration - expand up to full width
            for pair in node_weight_pair_list:
                node = pair[0]
                nodes.append(node)
                nodes_kept.append(node)
                select_num.append(num)
        else:
            for pair in node_weight_pair_list:
                node = pair[0]
                nodes.append(node)
                num = num_list[node.get_dvts_id()] # get num remaining per subtree
                if node.id in keep_node_ids:
                    nodes_kept.append(node)
                    select_num.append(num)
                else:
                    select_num.append(0)

        # get KV cache size
        kv_size = self.get_kv_size(nodes)
        with lock:
            global global_kv_size
            global_kv_size += kv_size

        return nodes, select_num

    def select_softmax_costmodel(self, node_list, node_weights, width, depth):

        # compute leaf mapping
        def compute_branch_leaf_mapping(leaf_list):
            branch_leaf_mapping = {}

            for leaf in leaf_list:
                current_node = leaf
                while current_node.parent is not None:  # Traverse up to the root
                    parent = current_node.parent
                    if parent.id not in branch_leaf_mapping:
                        branch_leaf_mapping[parent.id] = []
                    branch_leaf_mapping[parent.id].append(leaf.id)
                    current_node = parent  # Move up to the parent

            # Ensure leaf IDs are unique in each branch's list
            for branch_id in branch_leaf_mapping:
                branch_leaf_mapping[branch_id] = list(set(branch_leaf_mapping[branch_id]))

            return branch_leaf_mapping

        # compute mapping
        branch_leaf_mapping = compute_branch_leaf_mapping(node_list)

        # Decision variables
        N = len(node_list) # num leafs
        M = len(branch_leaf_mapping) # num branches
        x = [LpVariable(f"x_{i}", cat="Binary") for i in range(N)]  # Binary decision for each leaf
        y = [LpVariable(f"y_{j}", cat="Binary") for j in range(M)]  # Binary decision for each branch

        # Objective function: maximize outcome scores plus costs
        if self.lambdac == 0: # guard
            lambdac = 1e-4
        else:
            lambdac = -self.lambdac

        # build a map here of leaf node id -> index of binary variable
        leafnodemap = {}
        for i in range(N):
            leafnodemap[node_list[i].id] = i

        # build a map here of branch node id -> index of binary variable
        branchnodemap = {}
        idx = 0
        for j, leaf_nodes in branch_leaf_mapping.items():
            branchnodemap[j] = idx
            idx += 1

        # Define the problem
        problem = LpProblem("Tree_Selection_Problem", LpMaximize)
        Cost = [1/(M+N) for i in range(M+N)] # for now, set cost as 1/num_branch_nodes for each, later set it based on num_tokens

        # use num_retained_trajectories as normalized outcome score
        node_weight_pair_list = [(i, node, weight) for i, (node, weight) in enumerate(zip(node_list, node_weights))]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda triplet: triplet[2], reverse=True)
        weights = []
        for triplet in sorted_node_weight_pair_list:
            weights.append(triplet[2])
        weights = torch.tensor(weights)
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        outcome_score = []
        width_tmp = width
        for weight in exp_weights:
            if sum_exp_weights > 0:
                num = int(math.ceil(width_tmp * weight / sum_exp_weights))
                outcome_score.append(num)
                width_tmp -= num
                sum_exp_weights -= weight
            else:
                outcome_score.append(0)

        score_in_original_order = [0] * len(node_weights)
        for i, (orig_idx, node, weight) in enumerate(sorted_node_weight_pair_list):
            score_in_original_order[orig_idx] = outcome_score[i]
        O = [score_in_original_order[i] / sum(score_in_original_order) for i in range(len(score_in_original_order))]

        # Collect sentences from node_list
        sentences = []
        for i in range(N):
            if node_list[i].parent is not None:
                prevlen = len(node_list[i].parent.get_text())
            else:
                prevlen = 0
            s = node_list[i].get_text()
            s = s[prevlen:]
            sentences.append(s)

        # Compute the number of unique sequences
        unique_sequences = set(sentences)
        num_unique = len(unique_sequences)

        # if similarity
        lambdas = self.lambdas
        apply_sim = lambdas > 0 and N > 1 and num_unique > 1
        if apply_sim:
            # produce sequence embeddings here
            embeddings = self.model.encode(sentences, batch_size=64)
            embeddings = np.array(embeddings)

            # clustering sequences
            Z = linkage(embeddings, method='average', metric='cosine')
            clusters = fcluster(Z, 0.05, criterion='distance') - 1 # subtract to get 0-indexed labels
            K = len(np.unique(clusters))

            # link coverage variables to cluster selections
            coverage = [LpVariable(f"coverage_{i}", cat="Binary") for i in range(K)]  # Binary decision for each cluster
            for k in range(K):
                cluster_indices = [i for i, c in enumerate(clusters) if c == k]
                problem += lpSum(x[i] for i in cluster_indices) >= coverage[k], f"Coverage_Lower_{k}"

            # divide lambdas by num_clusters to scale term appropriately
            lambdas /= K

            # objective function
            problem += (
                lpSum(O[i] * x[i] for i in range(N)) +  # Maximize outcomes for selected leaves
                lpSum(lambdac * Cost[j] * y[j] for j in range(M)) +  # Include costs for branch nodes
                lpSum(lambdac * Cost[j+M] * x[j] for j in range(N)) + # Include costs for leaf nodes
                lpSum(lambdas * coverage[k] for k in range(K)) # include similarity enforcement
            ), "Objective"

        else:

            # objective function
            problem += (
                lpSum(O[i] * x[i] for i in range(N)) +  # Maximize outcomes for selected leaves
                lpSum(lambdac * Cost[j] * y[j] for j in range(M)) +  # Include costs for branch nodes
                lpSum(lambdac * Cost[j+M] * x[j] for j in range(N)) # Include costs for leaf nodes
            ), "Objective"

        # Constraints - Link y (branch selected) to x (leaf nodes in the branch)
        for j, leaf_nodes in branch_leaf_mapping.items():
            branch_node_idx = branchnodemap[j]
            leaf_nodes_mapped = [leafnodemap[i] for i in leaf_nodes]

            problem += y[branch_node_idx] <= lpSum(x[i] for i in leaf_nodes_mapped), f"Branch_{j}_activation"
            for i in leaf_nodes_mapped:
                problem += y[branch_node_idx] >= x[i], f"Branch_{j}_leaf_{i}_link"

        # constraint to not prune all nodes
        problem += lpSum(x[i] for i in range(N)) >= 1, "Keep_Node_Constraint"

        # Solve the problem
        problem.solve(PULP_CBC_CMD(msg=0))

        # decide which nodes to keep
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        weights = []
        idx = 0
        T = self.paras["softmax_temperature"]
        for pair in node_weight_pair_list:
            keep_node = x[idx].value() == 1
            if keep_node:
                weights.append(pair[1] / T)
            else:
                weights.append(float('-inf'))
            idx += 1

        # re-apply rebase-style sampling
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()

        nodes = []
        sorted_weights = []
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            sorted_weights.append(pair[1])

        # compute rebase weighting
        weights = torch.tensor(sorted_weights)
        exp_weights = torch.exp(weights)
        sum_exp_weights = exp_weights.sum()
        select_num = []
        nodes_kept = []
        idx = 0
        for weight in exp_weights:
            if sum_exp_weights > 0:
                num = int(math.ceil(width * weight / sum_exp_weights))
                if num > 0:
                    nodes_kept.append(nodes[idx])
                select_num.append(num)
                width -= num
                sum_exp_weights -= weight
            else:
                select_num.append(0)
            idx += 1

        # get KV cache size
        kv_size = self.get_kv_size(nodes)
        with lock:
            global global_kv_size
            global_kv_size += kv_size

        return nodes, select_num

    def select_softmax(self, node_list, node_weights, width, depth):
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()
        nodes = []
        weights = []
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            weights.append(pair[1])
        weights = torch.tensor(weights)
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        select_num = []
        nodes_kept = []
        idx = 0
        for weight in exp_weights:
            num = int(math.ceil(width * weight / sum_exp_weights))
            if num > 0:
                nodes_kept.append(nodes[idx])
            select_num.append(num)
            width -= num
            sum_exp_weights -= weight
            idx += 1

        # get KV cache size
        kv_size = self.get_kv_size(nodes)
        with lock:
            global global_kv_size
            global_kv_size += kv_size

        return nodes, select_num

    def select_and_expand(self, depth):
        cand_node_list = []
        cand_node_weights = []
        for node in self.depth_nodes[depth]:
            if node.is_leaf() == True or node.get_cum_tokens() >= self.paras["max_tokens"]:
                self.remaining_width -= 1

                if self.remaining_width_dvts is not None and node.get_dvts_id() is not None:
                    dvts_id = node.get_dvts_id()
                    self.remaining_width_dvts[dvts_id] -= 1
            else:
                cand_node_list.append(node)
                cand_node_weights.append(node.get_score())

        if self.remaining_width <= 0 or cand_node_list == []:
            return False
        if self.paras["select_method"] ==  "beam_search":
            nodes, widths = self.select_beam_search(cand_node_list, cand_node_weights, self.remaining_width, depth)
        elif self.paras["select_method"] ==  "softmax_costmodel":
            nodes, widths = self.select_softmax_costmodel(cand_node_list, cand_node_weights, self.remaining_width, depth)
        elif self.paras["select_method"] ==  "dvts":
            nodes, widths = self.select_dvts(cand_node_list, cand_node_weights, self.remaining_width, depth)
        elif self.paras["select_method"] ==  "softmax":
            nodes, widths = self.select_softmax(cand_node_list, cand_node_weights, self.remaining_width, depth)
        elif self.paras["select_method"] == "softmax_with_truncate":
            nodes, widths = self.select_softmax_with_truncation(cand_node_list, cand_node_weights, self.remaining_width)
        else:
            assert(False)
        for expand_node, width in zip(nodes, widths):
            if width >= 1:
                self.expand(expand_node, width)

        return True


@function
def reward_guided_search(s, id, question, ground_truth_answer, paras, reward_host, multimodel, model_config):

    # special handling for instruction tuned llama model
    if paras["policy_model_type"] == "llama":
        system_prompt = "Cutting Knowledge Date: December 2023\nToday Date: 05 Jan 2025\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
        s += system(system_prompt)
        s += user(question)
    else:
        s += (question + '\n')

    tree = Tree(s, paras, reward_host, multimodel, model_config)
    depth = 0
    while True:
        tree.reset_running_list()
        continue_search = tree.select_and_expand(depth)
        if continue_search == False:
            break
        running_list = tree.get_running_list()
        for state, parent in running_list:
            tree.insert(state, parent)

        depth += 1
        if depth >= 25:
            break
    history_list = tree.get_history_list()
    total_tokens = 0
    for state in history_list:
        total_tokens += state.get_meta_info("step")["completion_tokens"]

    all_nodes = tree.get_nodes()
    answers = []
    nodes_info = []
    answer_store_path = paras["store_path"] + f"answer_q{id}.json"
    for node in all_nodes:
        if node.get_parent() is not None:
            parent_id = node.get_parent().get_id()
        else:
            parent_id = None
        nodes_info.append({
            "id": node.get_id(),
            "text": node.get_text(),
            "score": node.get_score(),
            "parent_id": parent_id,
            "depth": node.get_depth(),
            "leaf": node.is_leaf()
        })
        if node.is_leaf():
            step_scores = []
            last_node = node
            while last_node.get_depth() > 0:
                step_scores.append(last_node.get_score())
                last_node = last_node.get_parent()
            step_scores.reverse()
            answers.append({"text":node.get_text(), "step_scores":step_scores})


    answer_for_the_question = {"id":id, "question": question, "model_answer":answers, "ground_truth_answer": ground_truth_answer["answer"], "total_tokens":total_tokens}
    json.dump(answer_for_the_question, open(answer_store_path, "w"), indent=4)
    return answer_for_the_question



def search_worker(search_dict, lock, prompts, test_examples, paras, policy_host, reward_host):
    while True:
        q_id = None
        with lock:
            for key in search_dict:
                if search_dict[key] == False:
                    search_dict[key] = True
                    q_id = int(key)
                    break
        if q_id == None:
            break
        state = reward_guided_search.run(id=q_id, question=prompts[q_id], ground_truth_answer=test_examples[q_id], paras=paras, reward_host=RuntimeEndpoint(reward_host), backend=RuntimeEndpoint(policy_host))
        answer_for_the_question = state.ret_value
        return answer_for_the_question



def main(args):
    t1 = time.time()
    prompts, test_examples = get_prompts(args)
    with open(args.parameter_path ,'r', encoding='utf-8') as file:
        paras = yaml.safe_load(file)
    input_list_dict = []

    if args.embed_device is not None:
        device = "cuda:" + str(args.embed_device)
    else:
        device = "cpu"

    # initialize sentence transformer
    if paras.get("lambdas", 0) > 0:
        multimodel = SentenceTransformer('math-similarity/Bert-MLM_arXiv-MP-class_zbMath', device=device)
    else:
        multimodel = None

    # load model config for profiling
    url = args.policy_host + "/get_model_info"
    response = requests.get(url)
    response_json = response.json()
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(response_json["model_path"])

    for i, prompt in enumerate(prompts):
        input_list_dict.append({"id":i, "question":prompt, "ground_truth_answer":test_examples[i], "paras":paras, "reward_host":RuntimeEndpoint(args.reward_host), "multimodel": multimodel, "model_config": model_config})
    states = reward_guided_search.run_batch(input_list_dict, backend=RuntimeEndpoint(args.policy_host), num_threads=paras["num_threads"], progress_bar=True)

    results = []
    total_gen_tokens = 0
    for s in states:
        answer = s.ret_value
        total_gen_tokens += answer["total_tokens"]
        results.append(answer)

    json.dump(results, open(args.output_path, "w"), indent=4)

    # save nonzero stats
    output_dir = os.path.dirname(args.output_path)
    log_file_path = os.path.join(output_dir, 'stats.log')

    t2 = time.time()

    # open the log file in write mode
    with open(log_file_path, 'w') as log_file:
        global global_kv_size
        global global_num_model_calls
        global global_num_tokens_generated

        log_message = f'Total KV cache size: {global_kv_size}'
        print(log_message)
        log_file.write(log_message + '\n')

        log_message = f'Number of Model Calls: {global_num_model_calls}'
        print(log_message)
        log_file.write(log_message + '\n')

        log_message = f'Number of Tokens Generated: {global_num_tokens_generated}'
        print(log_message)
        log_file.write(log_message + '\n')

        log_message = f'Time: {t2-t1}'
        print(log_message)
        log_file.write(log_message + '\n')

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input_path', type=str, required=True)
    args_parser.add_argument('--output_path', type=str, required=True)
    args_parser.add_argument('--parameter_path', type=str, required=True)
    args_parser.add_argument('--policy_host', type=str)
    args_parser.add_argument('--reward_host', type=str)
    args_parser.add_argument('--num_samples', type=int, default=None)
    args_parser.add_argument('--embed_device', type=int, default=None)
    args = args_parser.parse_args()
    main(args)
