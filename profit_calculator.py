def calculate_profit(acres, price_per_quintal):
    avg_yield = 9  # quintals per acre
    total_yield = acres * avg_yield
    income = total_yield * price_per_quintal
    return total_yield, income
