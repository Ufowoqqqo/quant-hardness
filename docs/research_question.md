# Research question

## Primary question

When and why does quantization turn an easy graph-ANN query into a hard one?

The question is empirical. The repository does not assume that quantization
causes a distinct query-hardness phenomenon.

## Operational definitions to fix before baseline experiments

- **Easy query:** to be defined using a pre-registered per-query search-cost
  and quality criterion under exact-vector search.
- **Hard query:** to be defined using the same criterion and graph/search
  parameters under quantized-vector search.
- **Quantization damage:** to be defined as paired, per-query changes in search
  quality and search cost.
- **Comparable search:** exact and quantized variants use the same graph and
  search parameters unless graph construction is the experimental variable.

Any change to these definitions must be recorded in
`docs/experiment_log.md` before interpreting results under the new definition.

