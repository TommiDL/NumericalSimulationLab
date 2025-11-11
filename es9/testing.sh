## Example: Checking the return code and taking action

for i in {1..20}
do
    echo "0000 0000 0000 000$i" > seed.in
    ./ga.exe
    if [ $? -eq 0 ]; then
       echo "Command executed successfully!"
    else
        echo "Command failed with return code: $?"
    fi

done

### restore seed to 1
echo "0000 0000 0000 0001" > seed.in
