for d in ./*/ ; 
do 
    (
        cd "$d" && jupyter nbconvert $(ls | grep ipynb | grep notebook) --to markdown --output Readme.md 
    );
    #var = $( ls | grep ipynb)
    #ls #| grep "ipynb" ;
    #echo $var 
done