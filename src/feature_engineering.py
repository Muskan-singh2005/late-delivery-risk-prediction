def create_features(df):

    df['shipping_pressure'] = df['Order Item Quantity'] / (df['Days for shipment (scheduled)'] + 1)

    df['urgency_index'] = df['Order Item Quantity'] / (df['Days for shipment (scheduled)'] + 1)

    df['discount_impact'] = df['Order Item Discount Rate'] * df['Order Item Quantity']

    df['profit_ratio'] = df['Order Profit Per Order'] / (df['Sales'] + 1)

    # 🔥 Strong new features
    df['cost_per_item'] = df['Sales'] / (df['Order Item Quantity'] + 1)

    df['high_discount_flag'] = (df['Order Item Discount Rate'] > 0.3).astype(int)

    df['bulk_order_flag'] = (df['Order Item Quantity'] > 5).astype(int)

    return df