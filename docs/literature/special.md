# Special Models

These are models that I deem 'special' in the sense that they're not exactly what GPs were originally tended for but they have usefulness in other fields. I consider these algorithms in the category of "jacking around" (see Neal Lawrences lecture for the origin of the phrase). In my mind, these are algorithms that tackle a special case problem in the application realm. It's a bit difficult to verify how it does outside of the corner case due to no real comprehensive code-bases or benchmarks of GPs in real-world applications. 

## Derivative Constraints

* [Scaling GPR with Derivatives](https://arxiv.org/abs/1810.12283) - Eriksson et. al. (2018) | [Code]()
  
  > They use MVM methods to enable one to solve for functions and their derivatives at the same time at scale.
  