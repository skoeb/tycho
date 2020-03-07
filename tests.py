from collections import Counter
# --- test that no duplicate plant_id_eias in eightsixty ---
Counter(eightsixty['plant_id_eia']).most_common(1)

