import pandas as pd
import splink.duckdb.comparison_level_library as cll
import splink.duckdb.comparison_library as cl
from splink.duckdb.blocking_rule_library import block_on
from splink.duckdb.linker import DuckDBLinker

pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 150
df = pd.read_parquet("cora_corrected_all_with_embeddings.parquet")


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
    "blocking_rules_to_generate_predictions": [],
    "comparisons": [],
}


linker = DuckDBLinker(df, settings)
