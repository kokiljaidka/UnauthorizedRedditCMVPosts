This dataset accompanies the following paper:

> Jaidka, K., & Ahmed, S. (2026). How Far Did They Go? The Persuasive Tactics of Covert LLM Agents in a Discontinued Field Experiment. *Proceedings of the 3rd Workshop on Natural Language Processing for Political Sciences (NLP4PS at LREC-COLING 2026)*.

Cite us as:

@inproceedings{jaidka2026covert,
  author    = {Jaidka, Kokil and Ahmed, Saifuddin},
  title     = {How Far Did They Go? {T}he Persuasive Tactics of Covert
               {LLM} Agents in a Discontinued Field Experiment},
  booktitle = {Proceedings of the 3rd Workshop on Natural Language
               Processing for Political Sciences},
  series    = {NLP4PS at {LREC}-{COLING} 2026},
  year      = {2026},
}


# UnauthorizedRedditCMVPosts
Annotated dataset corresponding to Reddit r/changemyview (r/CMV) comments by covert LLM agents, enriched with span-level labels for identity positioning (targeting/adoption), alignment and authority moves, and eight cognitive heuristics (e.g., availability, base-rate neglect, confirmation bias).


We introduce a newly annotated dataset of online argumentative comments, enriched with structured discourse and cognitive annotations. The corpus consists of Reddit comments drawn from r/changemyview, a deliberative discussion forum in which users engage in structured argumentation. Each comment is preserved in full and annotated at the span level using a standardized schema implemented via controlled large language model (LLM) prompting and post-validation.

The dataset includes three complementary annotation layers:

#(1) Identity and Positioning Layer.
Comments are annotated for identity-related discourse, distinguishing between (a) identity targeting (references to the interlocutor’s demographic, political, or social identity) and (b) identity adoption (first-person claims of demographic, professional, or experiential identity). Each instance includes a functional label (e.g., alignment, challenge, credibility, experiential authority), enabling analysis of how identity cues structure argumentative positioning.

#(2) Rhetorical Strategy Layer.
We annotate alignment moves and authority moves. Alignment moves capture how a speaker positions themselves relative to the interlocutor’s stance (positive alignment such as concession or acknowledgment; negative alignment such as disagreement or reframing). Authority moves capture epistemic credibility claims and are categorized into credentials, experiential, institutional, forum-based, external references, and appeals to social expectations. This layer supports analysis of conciliatory versus adversarial framing and the strategic deployment of epistemic authority.

#(3) Cognitive Heuristics and Bias Layer.
Comments are annotated for eight well-established cognitive heuristics and biases derived from decision science and behavioral economics: Law of Small Numbers, Availability Heuristic, Representativeness Heuristic, Base-Rate Neglect, Attribute Substitution, Affect Heuristic, Confirmation Bias, and Illusion of Validity. Instances are coded at the span level, identifying where intuitive or narrative-based reasoning substitutes for statistical or evidentiary reasoning.

All annotations are provided in tab-separated format with (i) binary presence indicators, (ii) frequency counts per category, and (iii) structured JSON fields preserving span-level labels and justifications. The schema was operationalized through deterministic prompting (temperature = 0) and strict JSON validation to ensure format consistency.

This resource enables research at the intersection of computational argumentation, discourse analysis, cognitive bias modeling, and LLM evaluation. It supports tasks such as rhetorical strategy detection, identity-based persuasion modeling, heuristic reasoning identification, and analysis of deliberative quality in online discussions. The layered design further allows researchers to examine interactions between identity framing, epistemic authority claims, and cognitive bias cues in argumentative discourse.
