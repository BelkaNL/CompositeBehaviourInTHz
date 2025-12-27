name: Run Python Script and Upload Plots as Artifacts

on:
  push:
    branches:
      - main  # This triggers the workflow on push to the 'main' branch
  pull_request:
    branches:
      - main  # This triggers the workflow on pull requests to 'main'

jobs:
  run_script:
    runs-on: ubuntu-latest  # Use the latest Ubuntu runner

    steps:
    - name: Checkout code
      uses: actions/checkout@v2  # Checks out your code from the repository

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9  # Python version

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install numpy matplotlib  # Add any other dependencies your script needs

    - name: Run Python script to generate plots
      run: |
        python src/test.py  # Ensure this path points to your script that generates plots

    # Step 1: Commit and push the generated plots to GitHub
    - name: Commit and push plots
      run: |
        git config user.name "github-actions[bot]"  # Set GitHub Actions bot name
        git config user.email "github-actions[bot]@users.noreply.github.com"  # Set bot email
        git add plots/*  # Add the generated plots to git
        git commit -m "Add new plots generated on $(date)"  # Commit the new plots
        git push  # Push the changes back to GitHub
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}  # Automatically provided by GitHub for authentication

    # Step 2: Upload generated plots as GitHub artifacts
    - name: Upload plots as artifacts
      uses: actions/upload-artifact@v3  # Use latest supported version for artifact uploads
      with:
        name: plots
        path: plots/  # Path to the generated plots
