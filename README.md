# Social game API

## Example of expected API logic & sequencing (high-level)

1. monday morning -- get_energy_pricing for monday day (requires state space information)
    1. needs friday's energy consumption, and grid information (pre-stored)
2. monday night -- submit_energy_consumption for monday day
    1. energy consumption stored in database
3. monday night -- get_points_and_base_points
    1. Update model (can be done in parallel with calculating earned points for monday day).
    2. Use loaded energy consumption to update model parameters
    3. Store model parameters in database, along with ack_id
    4. Calculate earned points for monday day
    5. Depends on prices for monday day (already calculated monday morning, monday's energy, base_points)
4. tuesday morning - get energy_pricing for tuesday day.
    1. Load latest model parameters, use them to generate newest price signal for current day
    2. Store price signal in database
    3. Return price signal
