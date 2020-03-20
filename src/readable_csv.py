# Converts the rows of data, with the specified
# headers, into an easily human readable csv string.
def readable_csv(rows, headers, digits=None):
	if digits is not None:
		for i, row in enumerate(rows):
			for j, col in enumerate(row):
				if isinstance(col, float):
					rows[i][j] = round(col, digits)

	# First, stringify everything.
	rows = [[str(c) for c in row] for row in rows]

	# Get the width needed for each column and add two.
	widths = [max([len(r[col]) for r in rows]) for col in range(len(rows[0]))]
	widths = [max(w, len(headers[i])) + 2 for i, w in enumerate(widths)]

	def space(row):
		return [c.ljust(widths[i]) for i, c in enumerate(row)]

	result = '%s\n'%(','.join(space(headers)))

	for row in rows:
		result += '%s\n'%(','.join(space(row)))

	return result[:-1]