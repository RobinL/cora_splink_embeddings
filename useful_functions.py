def get_cosine_level(column_name, threshold=0.9):
    return {
        "sql_condition": (
            f"list_cosine_similarity({column_name}_embeddings_l, "
            f"{column_name}_embeddings_r) > {threshold}"
        ),
        "label_for_charts": f"cosine similarity > {threshold}",
    }


def get_cosine_comparison(column_name):
    return {
        "output_column_name": column_name,
        "comparison_levels": [
            cll.null_level(column_name),
            get_cosine_level(column_name, 0.95),
            cll.else_level(),
        ],
    }


settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        block_on(x),
    ],
    "comparisons": [
        get_cosine_comparison("author"),
    ],
}
