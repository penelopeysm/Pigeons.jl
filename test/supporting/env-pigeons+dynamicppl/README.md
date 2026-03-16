# Tests

## Running the tests locally

To activate the environment for DynamicPPL testing, from the root of the Pigeons repo, type:

```
julia 
include("test/supporting/with-dynamicppl.jl")
```

Then to setup for running tests in test folder type:

```
include("test/supporting/setup_dynamicppl.jl")
```

which will add all packages needed for testing.

Then to run a test, simply call:

```
include("test/TEST_FILE.jl")
```

or 

```
include("test/supporting/env-pigeons+dynamicppl/TEST_FILE.jl")
```

where TEST_FILE is the test you want to run.

Here are tests for DynamicPPL logdensity and gradient:

```
include("test/test_turing.jl")
include("test/test_BufferedAD.jl")
```
Note that for test_BufferedAD.jl, only DynamicPPL targets test set needed to pass.