import pandas as pd
import splink.duckdb.comparison_level_library as cll
import splink.duckdb.comparison_library as cl
from splink.duckdb.blocking_rule_library import block_on
from splink.duckdb.linker import DuckDBLinker

pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 150
df = pd.read_parquet("cora_corrected_all_with_embeddings.parquet")
df


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
            cll.exact_match_level(column_name, term_frequency_adjustments=True),
            get_cosine_level(column_name, 0.99),
            get_cosine_level(column_name, 0.95),
            get_cosine_level(column_name, 0.9),
            get_cosine_level(column_name, 0.8),
            get_cosine_level(column_name, 0.5),
            get_cosine_level(column_name, 0.2),
            cll.else_level(),
        ],
    }


settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        block_on("class"),
    ],
    "comparisons": [
        get_cosine_comparison("author"),
        get_cosine_comparison("title"),
        get_cosine_comparison("venue"),
        get_cosine_comparison("note"),
        cl.levenshtein_at_thresholds("year", [1, 2], term_frequency_adjustments=True),
        cl.levenshtein_at_thresholds("pages", [1, 3], term_frequency_adjustments=True),
        cl.levenshtein_at_thresholds(
            "publisher", [1, 3, 5], term_frequency_adjustments=True
        ),
        cl.levenshtein_at_thresholds(
            "address", [1, 3, 5], term_frequency_adjustments=True
        ),
    ],
    # "retain_intermediate_calculation_columns": True,
}

linker = DuckDBLinker(df, settings)

linker.estimate_probability_two_random_records_match([block_on("title")], recall=0.7)

linker.estimate_u_using_random_sampling(5e6)

linker.estimate_parameters_using_expectation_maximisation(
    block_on("title"), estimate_without_term_frequencies=True
)
linker.estimate_parameters_using_expectation_maximisation(
    block_on("pages"), estimate_without_term_frequencies=True
)
linker.estimate_parameters_using_expectation_maximisation(
    block_on("publisher"), estimate_without_term_frequencies=True
)
display(linker.match_weights_chart())

df_predict = linker.predict(threshold_match_probability=0.7)
df_predict.as_pandas_dataframe().head()

linker.query_sql(
    f"""
                 select * from {df_predict.physical_name}
where gamma_publisher = 3

                 """
)
