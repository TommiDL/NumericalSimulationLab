## Example: Checking the return code and taking action

for i in {1..20}
do
    echo "0000 0000 0000 000$i" > seed.in
    # change filenames
    echo "New run with seed = $i"
    sed -i -e 's/seed[^ ]*./seed'"$i"'.dat/g' INPUT/input.dat
    cat INPUT/input.dat
    ./ga.exe
    if [ $? -eq 0 ]; then
       echo "Command executed successfully!"
    else
        echo "Command failed with return code: $?"
    fi

done

### restore seed to 1
echo "0000 0000 0000 0001" > seed.in
