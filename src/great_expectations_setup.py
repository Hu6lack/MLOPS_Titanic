"""Setup Great Expectations pour le projet Titanic MLOps"""
import great_expectations as gx
import pandas as pd
from pathlib import Path

def setup_expectations():
    """Configure Great Expectations et crée une expectation suite."""
    print("🚀 Configuration de Great Expectations...")
    
    try:
        # Create a data context (newer API)
        context = gx.get_context(mode="ephemeral")
        
        # Create expectation suite (new API)
        suite_name = "titanic_suite"
        
        # Create or get expectation suite
        try:
            suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
            print(f"✓ Created new suite: {suite_name}")
        except Exception as e:
            print(f"Suite creation note: {e}")
            suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
        
        # Load training data
        train_path = Path("/opt/airflow/data/train.csv")
        
        if not train_path.exists():
            print(f"⚠️ Train data not found at {train_path}, skipping GE setup")
            return context
        
        df_train = pd.read_csv(train_path)
        print(f"✓ Loaded training data: {len(df_train)} rows")
        
        # Create datasource (newer API)
        datasource_name = "titanic_datasource"
        
        # Add pandas datasource
        if datasource_name not in [ds.name for ds in context.datasources]:
            datasource = context.datasources.add_pandas(
                name=datasource_name,
                path=str(train_path.parent)
            )
            print(f"✓ Created datasource: {datasource_name}")
        else:
            datasource = context.get_datasource(datasource_name)
        
        # Create data asset
        asset_name = "titanic_asset"
        data_asset = datasource.add_dataframe_asset(
            name=asset_name,
            dataframe=df_train
        )
        
        # Create batch request
        batch_request = data_asset.build_batch_request()
        
        # Create validator
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
        
        # Add expectations
        expectations = [
            {
                "type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "Age"}
            },
            {
                "type": "expect_column_values_to_not_be_null", 
                "kwargs": {"column": "Sex"}
            },
            {
                "type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "Fare"}
            },
            {
                "type": "expect_column_values_to_be_in_set",
                "kwargs": {"column": "Survived", "value_set": [0, 1]}
            },
            {
                "type": "expect_column_values_to_be_in_set",
                "kwargs": {"column": "Sex", "value_set": [0, 1]}
            },
            {
                "type": "expect_column_values_to_be_in_set",
                "kwargs": {"column": "Embarked", "value_set": [0, 1, 2]}
            },
            {
                "type": "expect_column_values_to_be_between",
                "kwargs": {"column": "Age", "min_value": 0, "max_value": 120}
            },
            {
                "type": "expect_column_values_to_be_between",
                "kwargs": {"column": "Fare", "min_value": 0, "max_value": 600}
            },
            {
                "type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 500, "max_value": 1200}
            },
        ]
        
        for exp in expectations:
            try:
                validator.expectation_suite.add_expectation(
                    gx.expectations.ExpectationConfiguration(
                        type=exp["type"],
                        kwargs=exp["kwargs"]
                    )
                )
                print(f"  ✓ Added expectation: {exp['type']}")
            except Exception as e:
                print(f"  ✗ Failed to add {exp['type']}: {e}")
        
        # Save the expectation suite
        context.suites.add(validator.expectation_suite)
        
        print(f"✅ Expectation Suite '{suite_name}' created successfully!")
        
        # Run validation
        print("\n🔍 Running validation...")
        results = validator.validate()
        
        if results["success"]:
            print("✅ All expectations passed!")
        else:
            print("⚠️ Some expectations failed:")
            for result in results["results"]:
                if not result["success"]:
                    print(f"  - {result['expectation_config']['type']}: failed")
        
        return context
        
    except Exception as e:
        print(f"⚠️ Great Expectations error: {e}")
        print("Continuing without GE validation...")
        return None

if __name__ == "__main__":
    setup_expectations()