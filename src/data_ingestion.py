import pandas as pd
from pathlib import Path
from src.utils.common import read_params

def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    source_path = params['data_ingestion']['source_path']
    output_file = params['data_ingestion']['output_file']

    # Mock Data Creation for demonstration purposes if the source CSV is missing
if not Path(source_path).exists():
    print(f"Warning: Mocking data as source file not found at {source_path}.")
    data = pd.DataFrame({
        # 1. Felsic Major Oxides (High SiO2/K2O, Low MgO/Fe)
        'SiO2': [72.0, 75.5, 71.8, 73.5, 70.5, 74.0],  # Felsic (High Silica)
        'Al2O3': [15.2, 14.9, 15.5, 15.0, 16.0, 14.5],
        'MgO': [0.5, 0.15, 0.7, 0.2, 0.9, 0.3],        # Mafic-poor (Low Magnesia)
        'K2O': [4.0, 5.5, 3.5, 4.8, 5.0, 6.2],         # Alkaline
        'Fe2O3': [1.5, 0.8, 2.0, 1.2, 2.5, 1.0],

        # 2. Key Li Tracer Elements (Must be present for ratio calculations)
        'Li': [45.0, 150.0, 300.0, 50.0, 250.0, 180.0], # Li content (poor to enriched)
        'Rb': [150.0, 350.0, 500.0, 200.0, 450.0, 400.0],
        'Cs': [3.0, 10.0, 20.0, 5.0, 15.0, 12.0],

        # 3. Required Contextual/Categorical Column
        'Source_Sheet': ['Himalaya', 'Tibet', 'Himalaya', 'Tibet', 'Himalaya', 'Tibet'],

        # 4. Tracking Column (Non-feature)
        'Sample': ['L-01', 'L-02', 'L-03', 'L-04', 'L-05', 'L-06']
    })
else:
    try:
        data = pd.read_csv(source_path)
        except FileNotFoundError:
            print(f"Error: Source file not found at {source_path}. Please check config.")
            return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Data Ingestion complete: {output_path}")

if __name__ == "__main__":
    main()
