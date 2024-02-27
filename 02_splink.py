import pandas as pd
import splink.duckdb.comparison_level_library as cll
import splink.duckdb.comparison_library as cl
from splink.duckdb.blocking_rule_library import block_on
from splink.duckdb.linker import DuckDBLinker

pd.options.display.max_columns = 100
df = pd.read_parquet("cora_corrected_3_rows_with_embeddings.parquet")

df["unique_id"] = df.reset_index().index


def get_cosine_level(column_name, threshold=0.9):
    return {
        "sql_condition": (
            f"list_cosine_similarity({column_name}_embedding_l, "
            f"{column_name}_embedding_r) > {threshold}"
        ),
        "label_for_charts": f"cosine similarity > {threshold}",
    }


def get_cosine_comparison(column_name):
    return {
        "output_column_name": column_name,
        "comparison_levels": [
            cll.null_level(column_name),
            cll.exact_match_level(column_name, term_frequency_adjustments=True),
            get_cosine_level(column_name, 0.9),
            get_cosine_level(column_name, 0.8),
            get_cosine_level(column_name, 0.5),
            get_cosine_level(column_name, 0.2),
            cll.else_level(),
        ],
    }


author = {
    "output_column_name": "Author",
    "comparison_levels": [
        cll.null_level("author"),
        cll.exact_match_level("author", term_frequency_adjustments=True),
        get_cosine_level("author", 0.9),
        get_cosine_level("author", 0.8),
        get_cosine_level("author", 0.5),
        cll.else_level(),
    ],
}

settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        block_on("class"),
        "1=1",
    ],
    "comparisons": [
        get_cosine_comparison("author"),
        get_cosine_comparison("title"),
        get_cosine_comparison("venue"),
        get_cosine_comparison("note"),
    ],
    "retain_intermediate_calculation_columns": False,
}

linker = DuckDBLinker(df, settings)
linker.predict().as_pandas_dataframe()
