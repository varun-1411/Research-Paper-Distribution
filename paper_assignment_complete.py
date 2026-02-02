"""
Paper Assignment Optimization - Complete Example

This script demonstrates how to use the paper assignment optimizer.
It includes a function to test with a single file or multiple files.

Requirements:
    pip install pandas openpyxl gurobipy

If you don't have a Gurobi license, you can use the free academic license
or modify this to use PuLP/CBC solver instead.
"""

import pandas as pd
import glob
import os
from gurobipy import Model, GRB, quicksum
import gurobipy as gp


def read_bid_file(filepath):
    """
    Read a single bid file and extract team info and bids.
    
    Handles the specific format where:
    - Row 2: Team name
    - Row 3-4: Member names  
    - Row 8+: Paper bids (Paper | Bid | Status)
    """
    df = pd.read_excel(filepath, header=None)
    
    # Extract team name
    team_name = str(df.iloc[1, 3]).strip() if pd.notna(df.iloc[1, 3]) else "Unknown"
    
    # Extract member names
    member1 = str(df.iloc[2, 3]).strip() if pd.notna(df.iloc[2, 3]) else ""
    member2 = str(df.iloc[3, 3]).strip() if pd.notna(df.iloc[3, 3]) else ""
    
    members = []
    if member1 and member1.lower() != 'nan':
        members.append(member1)
    if member2 and member2.lower() != 'nan':
        members.append(member2)
    members_str = ", ".join(members) if members else "No members"
    
    # Extract bids
    bids = {}
    for i in range(7, len(df)):
        paper = df.iloc[i, 2]
        bid = df.iloc[i, 3]
        
        if pd.isna(paper) or str(paper).strip().lower() == 'sum':
            break
            
        if pd.notna(bid):
            try:
                bids[str(paper).strip()] = float(bid)
            except ValueError:
                continue
    
    return {
        'team_name': team_name,
        'members': members_str,
        'bids': bids
    }


def assign_papers(teams_data):
    """
    Assign papers to teams using Gurobi optimization.
    
    Args:
        teams_data: List of dicts with keys 'team_name', 'members', 'bids'
                   where 'bids' is a dict {paper_name: bid_value}
    
    Returns:
        DataFrame with columns: Team Name, Team Members, Assigned Paper, Bid Value
    """
    # Get all unique papers
    all_papers = set()
    for team in teams_data:
        all_papers.update(team['bids'].keys())
    all_papers = sorted(list(all_papers))
    
    teams = [t['team_name'] for t in teams_data]
    
    # Build bid matrix
    bid_matrix = {}
    for team_data in teams_data:
        team = team_data['team_name']
        bid_matrix[team] = {paper: team_data['bids'].get(paper, 0) for paper in all_papers}
    
    # Create model
    # model = Model("PaperAssignment")
    env = gp.Env(params={
    "WLSACCESSID": "802dcbf8-cb86-427b-9671-66cde14a3711",
    "WLSSECRET": "68d3d0ec-70f7-4dbe-af3a-b7258b4f159e",
    "LICENSEID": 2544493,
})

    model = gp.Model("PaperAssignment", env=env)
    model.setParam('OutputFlag', 0)  # Suppress output
    
    # Variables
    x = {(team, paper): model.addVar(vtype=GRB.BINARY, name=f"x_{team}_{paper}")
         for team in teams for paper in all_papers}
    
    model.update()
    
    # Objective: maximize total bids
    model.setObjective(
        quicksum(bid_matrix[team][paper] * x[team, paper] 
                 for team in teams for paper in all_papers),
        GRB.MAXIMIZE
    )
    
    # Each team gets exactly one paper
    for team in teams:
        model.addConstr(quicksum(x[team, paper] for paper in all_papers) == 1)
    
    # Each paper goes to at most one team
    for paper in all_papers:
        model.addConstr(quicksum(x[team, paper] for team in teams) <= 1)
    
    # Solve
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        raise Exception(f"Optimization failed with status {model.status}")
    
    # Extract results
    results = []
    for team_data in teams_data:
        team = team_data['team_name']
        for paper in all_papers:
            if x[team, paper].X > 0.5:
                results.append({
                    'Team Name': team,
                    'Team Members': team_data['members'],
                    'Assigned Paper': paper,
                    'Bid Value': bid_matrix[team][paper]
                })
                break
    
    return pd.DataFrame(results), model.objVal


def run_assignment(bid_folder, output_csv="paper_assignments.csv"):
    """
    Main function to run paper assignment.
    
    Args:
        bid_folder: Path to folder containing bid_info_*.xlsx files
        output_csv: Path for output CSV file
    
    Returns:
        DataFrame with assignments
    """
    # Find all bid files
    files = glob.glob(os.path.join(bid_folder, "bid_info_*.xlsx"))
    if not files:
        files = glob.glob(os.path.join(bid_folder, "*.xlsx"))
    
    if not files:
        raise FileNotFoundError(f"No Excel files found in {bid_folder}")
    
    # Read all files
    teams_data = []
    for f in files:
        try:
            data = read_bid_file(f)
            teams_data.append(data)
            print(f"Loaded: {data['team_name']}")
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
    
    # Optimize
    result_df, total_value = assign_papers(teams_data)
    
    # Save
    result_df.to_csv(output_csv, index=False)
    print(f"\nTotal bid satisfaction: {total_value}")
    print(f"Results saved to: {output_csv}")
    
    return result_df


# Example usage with sample data (for testing without files)
def demo_with_sample_data():
    """Demo with sample data to show how the optimizer works."""
    # Sample teams data
    teams_data = [
        {
            'team_name': 'Radhika',
            'members': 'Shreya Ghosh',
            'bids': {'wang2019': 10, 'thomas2018': 90, 'orda1990': 0, 
                    'loachim1998': 0, 'kilein2005': 0, 'fakcharoenphol2006': 0,
                    'eppstein1998': 0, 'ding2008': 0, 'dean2004': 0, 'ahmadi2025': 0}
        },
        {
            'team_name': 'Team_Alpha',
            'members': 'John Doe, Jane Smith',
            'bids': {'wang2019': 50, 'thomas2018': 20, 'orda1990': 30, 
                    'loachim1998': 0, 'kilein2005': 0, 'fakcharoenphol2006': 0,
                    'eppstein1998': 0, 'ding2008': 0, 'dean2004': 0, 'ahmadi2025': 0}
        },
        {
            'team_name': 'Team_Beta',
            'members': 'Alice Brown',
            'bids': {'wang2019': 25, 'thomas2018': 25, 'orda1990': 50, 
                    'loachim1998': 0, 'kilein2005': 0, 'fakcharoenphol2006': 0,
                    'eppstein1998': 0, 'ding2008': 0, 'dean2004': 0, 'ahmadi2025': 0}
        }
    ]
    
    print("Running demo with sample data...")
    result_df, total = assign_papers(teams_data)
    print("\nAssignments:")
    print(result_df.to_string(index=False))
    print(f"\nTotal bid satisfaction: {total}")
    return result_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run with provided folder
        folder = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "paper_assignments.csv"
        run_assignment(folder, output)
    else:
        print("Usage: python paper_assignment_complete.py <bid_folder> [output.csv]")
        print("\nRunning demo with sample data instead...")
        demo_with_sample_data()