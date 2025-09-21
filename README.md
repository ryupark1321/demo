Change permissions of the scripts
```
chmod 755 check_env.sh
chmod 755 runner.sh
```

To run the scripts
```
./check_env.sh
./runner.sh
```

runner.sh will create `demo.txt` and `demo_safe.txt`.
CUDA file -> Output mapping
`task.cu` -> `demo_safe.txt`
`task_unsafe.cu` -> `demo.txt`
