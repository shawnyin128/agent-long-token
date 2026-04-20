# Project Proposal

## Problem

In multi-agent LLM systems, agents communicate over multiple rounds to collaboratively solve a task. Each agent's prompt must carry the full communication history, so prompt length grows as $O(N \times R \times L)$ where $N$ is agent count, $R$ is rounds, and $L$ is average message length. This raises inference cost and introduces noise from redundant or outdated content.

We observe that this history is not homogeneous — it contains **redundancy** (multiple agents restating the same point), **superseded information** (early claims later corrected), and **convergence overhead** (full debate retained after consensus). Yet existing compression methods treat prompts as flat token sequences, and training-based approaches require parameter updates. No prior work has systematically analyzed *what* accumulates in multi-agent communication and *why*.

## Prior Work

Token-level compression (Selective Context, LLMLingua) reduces prompt length but ignores inter-message structure. OPTIMA trains agents to communicate concisely via SFT/DPO but requires fine-tuning. AgentDiet removes waste from single-agent trajectories. None analyze the structure of multi-agent communication itself — our work fills this gap. We adopt the multi-agent debate setting from prior work but focus on understanding communication structure rather than optimizing it through training.

## Approach

Our pipeline is LLM-only throughout and consists of two parts.

**P0: Structural Analysis.** We use an LLM to decompose each agent message into atomic, self-contained claims, each tagged with source agent and round number. We then embed all claims with a sentence encoder and cluster them by semantic topic via HDBSCAN. Within each cluster, claims are ordered by round to form temporal chains that show how a topic evolved — from initial proposals, through disagreements, to consensus or abandonment. We analyze the resulting structure along several dimensions: claim type taxonomy (proposals, evidence, corrections, agreements), distribution across agents and rounds, convergence patterns per cluster, and the fraction of claims that are redundant with or superseded by later content.

**P1 (Stretch): ICL-Guided Communication.** If P0 reveals consistent waste patterns — e.g., agents spending many tokens restating settled points — we design in-context learning prompts that guide agents toward more efficient communication from the start, reducing waste at the source rather than compressing it after the fact.

## Data & Plan

**Dataset.** We sample questions from MMLU across multiple subject areas. For each question, we run a multi-agent debate with 3 LLM agents (open-source, locally deployed) over 5–10 rounds, collecting the full dialogue logs. This produces 100–200 multi-agent conversations as the basis for structural analysis. All components — agents, claim extraction, and embedding — use LLMs or LLM-derived models exclusively.

**Evaluation.** Three questions:
- What fraction of tokens in multi-agent communication is redundant or superseded?
- Can claim-level clustering effectively capture the topic structure of the conversation? (NMI/ARI against human annotations on ~30 dialogues.)
- Can ICL guidance reduce token usage while maintaining accuracy? (Compared against vanilla, sliding-window, and summarization baselines.)

**Timeline.** Wk 1–2: pipeline setup, MMLU sampling, dialogue collection, claim extraction. Wk 3–4: clustering, annotation, structural analysis. Wk 5: ICL experiments. Wk 6: write-up.

---

## Professor's Feedback (received before project start)

> I think the general topic is good here but I have concerns about approach P0. I would encourage you to work through an example of this: what are the claims you get from the model in a setting like MMLU? What does the clustering actually achieve?
>
> Often, multi-agent debate doesn't achieve that much gain over a single-agent approach, so my worry is that all of the heavyweight analysis in P0 will mostly be dissecting stuff that doesn't improve the accuracy. I would perhaps focus on a task where multi-agent debate gives the most robust gains, then try to really understand where those gains come from, then see how you can design a compression scheme around that.
