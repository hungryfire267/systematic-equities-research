import datetime as dt   
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from pathlib import Path
from scripts.run_fetch import ASXPipeline


@pytest.fixture(scope="module")
def get_dates() -> tuple[dt.datetime, dt.datetime]: 
    start_date = dt.datetime(2026, 1, 11)
    end_date = start_date + dt.timedelta(days=6)
    return start_date, end_date

@pytest.fixture(scope="module")
def company_data() -> pd.DataFrame:
    companies_df = pd.DataFrame({ 
        "asxCode": ["CBA", "BHP.AX", "ZIP.AX"],
        "companyName": ["COMMONWEALTH BANK OF AUSTRALIA.", "BHP GROUP LIMITED", "ZIP CO LIMITED.."],
        "industry": ["Banks", "Materials", "Financial Services"]
    })
    return companies_df

@pytest.fixture(scope="module")
def run_pipeline(company_data, get_dates) -> ASXPipeline:
    companies_df = company_data
    start_date, end_date = get_dates
    pipeline = ASXPipeline(companies_df, start_date, end_date)
    market_cap = pipeline.GetMarketCap()
    company_data_dict, asx_data_dict = pipeline.GetData(market_cap)
    return pipeline, company_data_dict, asx_data_dict

def bundle(types: str, get_dates, run_pipeline) -> tuple[dict[str, str], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: 
    validate_type(types)
    pipeline, memory_company_dict, memory_asx_dict = run_pipeline
    expected_company_dict = expected_company_dicts(get_dates)
    expected_asx_dict = expected_asx_dicts(get_dates)
    if (types == "company"): 
        return pipeline.company_paths_dict, memory_company_dict, expected_company_dict
    else: 
        return pipeline.asx_paths_dict, memory_asx_dict, expected_asx_dict
    
def validate_type(types: str) -> None: 
    if types not in {"company", "asx"}:
        raise ValueError("Invalid type. Type must be either 'company' or 'asx'.")

@pytest.mark.parametrize("types", ["company", "asx"])
def test_validate_type(types: str) -> None: 
    validate_type(types)

@pytest.mark.parametrize("types", ["invalid", 123, None, ""])
def test_validate_type_invalid(types: str): 
    with pytest.raises(ValueError, match="Invalid type"): 
        validate_type(types)

def expected_company_dicts(get_dates) -> dict[str, pd.DataFrame]: 
    start_date, end_date = get_dates
    expected_price_df = pd.DataFrame({
        "Date": pd.date_range(start=start_date, end=end_date, freq="B"),
        "CBA.AX": [154.080002, 154.820007, 152.880005, 153.500000, 154.300003], 
        "BHP.AX": [46.509998, 47.580002, 48.119999, 49.369999, 48.990002],
        "ZIP.AX": [3.55, 3.28, 3.28, 3.08, 3.08]
    })
    
    expected_volume_df = pd.DataFrame({ 
        "Date": pd.date_range(start=start_date, end=end_date, freq="B"),
        "CBA.AX": [1085692, 1324215, 1923233, 1715980, 1875372],
        "BHP.AX": [11129168, 8959090, 7244698, 14145071, 12369669],
        "ZIP.AX": [21310174, 26507935, 11412164, 21895218, 12750184]                   
    })
    
    expected_returns_df = pd.DataFrame({
        "Date": pd.date_range(start=start_date, end=end_date, freq="B"), 
        "CBA.AX": [0.005613, 0.004803, -0.012531,  0.004055, 0.005212], 
        "BHP.AX": [-0.025356, 0.023006, 0.011349, 0.025977, -0.007697],
        "ZIP.AX": [-0.002809, -0.076056, 0.0, -0.06, 0.0]
    })
    
    expected_log_returns_df = pd.DataFrame({
        "Date": pd.date_range(start=start_date, end=end_date, freq="B"), 
        "CBA.AX": [0.005597, 0.004791, -0.012610, 0.004047, 0.005198], 
        "BHP.AX": [-0.025683, 0.022745, 0.011285, 0.025645, -0.007727], 
        "ZIP.AX": [-0.002813, -0.079104, 0.0, -0.062914, 0.0]
    })
    
    expected_market_cap_df = pd.DataFrame({
        "Date": pd.date_range(start_date, end=end_date, freq="B"),
        "CBA.AX": [2.576408e+11,  2.588782e+11, 2.556343e+11, 2.566710e+11, 2.580087e+11],
        "BHP.AX": [2.361876e+11, 2.416213e+11, 2.443635e+11, 2.507113e+11, 2.487816e+11],
        "ZIP.AX": [4.510927e+09, 4.167842e+09, 4.167842e+09, 3.913705e+09, 3.913705e+09]
    })
    
    return {
        "prices": expected_price_df, 
        "volume": expected_volume_df, 
        "returns": expected_returns_df, 
        "log_returns": expected_log_returns_df, 
        "market_cap": expected_market_cap_df
    }
    
def expected_asx_dicts(get_dates) -> dict[str, pd.DataFrame]:
    start_date, end_date = get_dates
    expected_asx_prices_df = pd.DataFrame({ 
        "Date": pd.date_range(start=start_date, end=end_date, freq="B"),
        "^AXJO": [8759.400391, 8808.500000, 8820.599609, 8861.700195, 8903.900391]
    })
    
    expected_asx_returns_df = pd.DataFrame({
        "Date": pd.date_range(start=start_date, end=end_date, freq="B"),
        "^AXJO": [0.004772, 0.005605, 0.001374,  0.004660, 0.004762]
    })
    
    expected_asx_log_returns_df = pd.DataFrame({ 
        "Date": pd.date_range(start_date, end=end_date, freq="B"),
        "^AXJO": [ 0.004761, 0.005590, 0.001373, 0.004649, 0.004751]
    })
    
    return { 
        "prices": expected_asx_prices_df, 
        "returns": expected_asx_returns_df, 
        "log_returns": expected_asx_log_returns_df
    }


@pytest.mark.parametrize("types", ["company", "asx"])
def test_GetData_files_exist(types: str, get_dates, run_pipeline) -> None: 
    paths_dict, _, _ = bundle(types, get_dates, run_pipeline)
    for key, path in paths_dict.items():
        assert Path(path).exists(), f"{key} file does not exist at {path}"
        
@pytest.mark.parametrize("types", ["company", "asx"])
def test_GetData_disk_memory_match(types: str, get_dates, run_pipeline) -> None: 
    paths_dict, memory_dict, _ = bundle(types, get_dates, run_pipeline) 
    for key, path in paths_dict.items(): 
        df_disk = pd.read_parquet(path, engine = "pyarrow")
        df_memory = memory_dict[key]
        assert_frame_equal(df_disk, df_memory, check_exact=False, rtol=1e-5), f"{key} data in memory does not match data on disk"
        
@pytest.mark.parametrize("types", ["company", "asx"])
def test_GetData_disk_expected_match(types: str, get_dates, run_pipeline) -> None: 
    paths_dict, memory_dict, expected_dict = bundle(types, get_dates, run_pipeline)
    for key, path in paths_dict.items():
        df_disk = pd.read_parquet(path, engine = "pyarrow")
        assert_frame_equal(df_disk, expected_dict[key], check_exact=False, rtol=1e-5), f"{key} data on disk does not match expected data"





