#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 09:39:37 2024

@author: Mónica Sagastuy-Breña
"""
import pandas as pd
# Variables for new maximum_size and minimum_size values
new_maximum_size = 200  # Replace with your calculated value
new_minimum_size = 50   # Replace with your calculated value
iteration_year = '2030'

# Load the CSV file into a DataFrame
file_path = 'parameters.csv'  # Path to the uploaded file
df = pd.read_csv(file_path)

# Find row that has description column value equal to 2030
id_row = df.index[df['description'] == iteration_year].tolist()
df.at[id_row[0], 'minimum_size'] = new_minimum_size
df.at[id_row[0], 'maximum_size'] = new_maximum_size

# Optionally, save the updated DataFrame back to a CSV file
df.to_csv('updated_parameters.csv', index=False)