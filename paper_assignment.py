"""
Paper Assignment Optimization using Gurobi

Reads all bid files from 'bids_folder' and assigns papers to teams
based on maximizing total bid satisfaction.

Requirements:
    pip install pandas openpyxl gurobipy

Usage:
    python paper_assignment.py
    
Output:
    - paper_assignments.csv: Assignment results
    - paper_assignment.lp: LP formulation file
"""

import pandas as pd
import glob
import os
from gurobipy import Model, GRB, quicksum


def read_bid_file(filepath):
    """
    Read a single bid file and extract team info and bids.
    
    Expected format:
    - Row 2 (index 1): Team name in column D (index 3)
    - Row 3 (index 2): Member 1 name in column D (index 3)
    - Row 4 (index 3): Member 2 name in column D (index 3)
    - Rows 8+ (index 7+): Paper names in column C (index 2), Bids in column D (index 3)
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
    members_str = ", ".join(members) if members else "No members listed"
    
    # Extract bids
    bids = {}
    for i in range(7, len(df)):
        paper = df.iloc[i, 2]
        bid = df.iloc[i, 3]
        
        # Stop at "Sum" row or invalid entries
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


def load_all_bids(folder_path):
    """Load all bid files from bids_folder."""
    # Find all Excel files
    patterns = [
        os.path.join(folder_path, "bid_info_*.xlsx"),
        os.path.join(folder_path, "*.xlsx"),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    # Remove duplicates
    files = list(dict.fromkeys(files))
    
    if not files:
        raise FileNotFoundError(f"No Excel files found in {folder_path}")
    
    teams_data = []
    for filepath in files:
        try:
            data = read_bid_file(filepath)
            teams_data.append(data)
            print(f"Loaded: {data['team_name']} - Members: {data['members']} ({len(data['bids'])} papers)")
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
    
    return teams_data


def optimize_assignment(teams_data, lp_filename="paper_assignment.lp"):
    """
    Solve the paper assignment problem using Gurobi.
    
    Maximize: sum of bid values for assigned papers
    Subject to:
        - Each team gets exactly one paper
        - Each paper is assigned to at most one team
    
    Also writes the LP formulation to a file.
    """
    # Collect all unique papers
    all_papers = set()
    for team in teams_data:
        all_papers.update(team['bids'].keys())
    all_papers = sorted(list(all_papers))
    
    teams = [t['team_name'] for t in teams_data]
    
    print(f"\n{'='*60}")
    print(f"Optimization Setup:")
    print(f"  - Number of teams: {len(teams)}")
    print(f"  - Number of papers: {len(all_papers)}")
    print(f"{'='*60}")
    
    # Create bid matrix
    bid_matrix = {}
    for team_data in teams_data:
        team = team_data['team_name']
        bid_matrix[team] = {paper: team_data['bids'].get(paper, 0) for paper in all_papers}
    
    # Create Gurobi model
    model = Model("PaperAssignment")
    model.setParam('OutputFlag', 0)  # Suppress solver output
    
    # Decision variables: x[team, paper] = 1 if team is assigned paper
    # Using cleaner variable names for LP file readability
    x = {}
    for team in teams:
        for paper in all_papers:
            # Clean variable name (remove special characters)
            clean_team = team.replace(" ", "_").replace("-", "_")
            clean_paper = paper.replace(" ", "_").replace("-", "_")
            var_name = f"assign_{clean_team}_{clean_paper}"
            x[team, paper] = model.addVar(vtype=GRB.BINARY, name=var_name)
    
    model.update()
    
    # Objective: Maximize total bid value
    model.setObjective(
        quicksum(bid_matrix[team][paper] * x[team, paper] 
                 for team in teams for paper in all_papers),
        GRB.MAXIMIZE
    )
    
    # Constraint 1: Each team gets exactly one paper
    for team in teams:
        clean_team = team.replace(" ", "_").replace("-", "_")
        model.addConstr(
            quicksum(x[team, paper] for paper in all_papers) == 1,
            name=f"OnePerTeam_{clean_team}"
        )
    
    # Constraint 2: Each paper assigned to at most one team
    for paper in all_papers:
        clean_paper = paper.replace(" ", "_").replace("-", "_")
        model.addConstr(
            quicksum(x[team, paper] for team in teams) <= 1,
            name=f"OnePerPaper_{clean_paper}"
        )
    
    model.update()
    
    # Write LP file BEFORE solving
    model.write(lp_filename)
    print(f"\nLP file written to: {lp_filename}")
    
    # Solve
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal solution found!")
        print(f"Total bid satisfaction: {model.objVal:.1f}")
        
        # Extract assignments
        assignments = []
        for team_data in teams_data:
            team = team_data['team_name']
            for paper in all_papers:
                if x[team, paper].X > 0.5:
                    bid_value = bid_matrix[team][paper]
                    assignments.append({
                        'Team Name': team,
                        'Team Members': team_data['members'],
                        'Assigned Paper': paper,
                        'Bid Value': bid_value
                    })
                    break
        
        return assignments, model.objVal
    
    elif model.status == GRB.INFEASIBLE:
        print("Error: Problem is infeasible (more teams than papers?)")
        return None, None
    else:
        print(f"Optimization ended with status: {model.status}")
        return None, None


def main():
    # Configuration - change these paths if needed
    BIDS_FOLDER = "bids_folder"
    OUTPUT_CSV = "paper_assignments.csv"
    OUTPUT_LP = "paper_assignment.lp"
    
    print("="*60)
    print("Paper Assignment Optimization")
    print("="*60)
    
    # Check if folder exists
    if not os.path.isdir(BIDS_FOLDER):
        print(f"Error: Folder '{BIDS_FOLDER}' not found.")
        print(f"Please ensure 'bids_folder' is in the same directory as this script.")
        return
    
    # Load all bid files
    print(f"\nLoading bid files from '{BIDS_FOLDER}'...")
    teams_data = load_all_bids(BIDS_FOLDER)
    
    if len(teams_data) == 0:
        print("Error: No valid bid files found.")
        return
    
    # Run optimization (also writes LP file)
    print("\nRunning optimization...")
    assignments, obj_val = optimize_assignment(teams_data, lp_filename=OUTPUT_LP)
    
    if assignments:
        # Save results
        result_df = pd.DataFrame(assignments)
        result_df.to_csv(OUTPUT_CSV, index=False)
        
        # Print results
        print(f"\n{'='*60}")
        print("Assignment Results")
        print("="*60)
        print(result_df.to_string(index=False))
        print(f"\nTotal Bid Satisfaction: {obj_val:.1f}")
        print(f"\nResults saved to: {OUTPUT_CSV}")
        print(f"LP formulation saved to: {OUTPUT_LP}")
    else:
        print("\nOptimization failed.")


if __name__ == "__main__":
    main()