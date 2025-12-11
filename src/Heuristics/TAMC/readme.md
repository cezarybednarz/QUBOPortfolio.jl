## Running TAMC 

Running https://github.com/USCqserver/tamc as a subprocess.

# Example usage:

```
run_tamc("../tamc/examples/easy_method_pt.yml", "../tamc/examples/easy_instance.txt", "easy_results.yml")
result = read_output_yaml("easy_results.yml")
println(result)
```

this corresponds to the following CLI command:
```
./tamc/target/release/tamc tamc/examples/easy_method_pt.yml tamc/examples/easy_instance.txt easy_results.yml
```
