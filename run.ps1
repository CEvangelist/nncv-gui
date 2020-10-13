
# Requires Anaconda3 installed in X:\Users\<your_user_name>

powershell.exe -ExecutionPolicy ByPass -WindowStyle Hidden -Command `
"& '$env:USERPROFILE\Anaconda3\shell\condabin\conda-hook.ps1' ;
 conda activate '$env:USERPROFILE\Anaconda3' ;
 python main.py"
