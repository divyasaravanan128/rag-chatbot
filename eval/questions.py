# eval/questions.py
# Ground truth answers written from your actual document

eval_pairs = [
    # ── Specific lookups — tests retrieval precision ──────────────────────
    {
        "question": "What are the warranty terms for Aurora wallpaper?",
        "ground_truth": "2 year warranty. Does not cover damage caused by misuse, negligence, or normal wear and tear."
    },
    {
        "question": "What is the product name listed under Annexure 1C?",
        "ground_truth": "Aurora - 4011-3 Green 3D Geometric Wallpaper"
    },
    {
        "question": "Who is the vendor mentioned in Annexure 1C?",
        "ground_truth": "Excelhome"
    },
    {
        "question": "What is the average utilization area for Wallpaper 771088?",
        "ground_truth": "50 square feet"
    },
    {
        "question": "What inspection requirement is mentioned at handover?",
        "ground_truth": "Inspection to be done by customer during handover. Issues, if any, will be rectified."
    },

    # ── Summary questions — tests context recall ───────────────────────────
    {
        "question": "Summarise the warranty coverage and exclusions for the traded goods.",
        "ground_truth": "Warranty is 2 years. Coverage excludes damage from misuse, negligence, and normal wear and tear. Customer must inspect at handover."
    },
    {
        "question": "What product category does the Aurora wallpaper fall under?",
        "ground_truth": "Aurora Wallpaper"
    },
    {
        "question": "What happens if issues are found during handover inspection?",
        "ground_truth": "Issues found during handover inspection will be rectified."
    },

    # ── Out-of-scope questions — tests faithfulness (should say 'I don't know') ──
    {
        "question": "What is the price of the Aurora wallpaper?",
        "ground_truth": "The document does not mention the price of the Aurora wallpaper."
    },
    {
        "question": "What is the return policy if the product is damaged during shipping?",
        "ground_truth": "The document does not mention a return policy for shipping damage."
    },
    {
        "question": "Who is the contact person for warranty claims?",
        "ground_truth": "The document does not mention a contact person for warranty claims."
    },

    # ── Reasoning question — tests answer relevancy ───────────────────────
    {
        "question": "Does the warranty cover a wallpaper that peeled off after 18 months of normal use?",
        "ground_truth": "Normal wear and tear is excluded from the warranty. Peeling after 18 months of normal use would likely not be covered."
    },
]