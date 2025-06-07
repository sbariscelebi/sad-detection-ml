sad_corr = df[numerical_cols].corrwith(df['SAD'], method='pearson')
plt.savefig("correlation_matrix.svg", format='svg', bbox_inches='tight', pad_inches=0.1)