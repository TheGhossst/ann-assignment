for ($i = 0; $i -le 9; $i++) {
    Write-Host "Running image $i.png"
    python main.py test --image ./images/$i.png
}