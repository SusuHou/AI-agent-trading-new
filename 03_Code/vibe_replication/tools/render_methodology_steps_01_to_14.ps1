$ErrorActionPreference = 'Stop'

$inputDocx = 'C:\Users\housu\Documents\AI agent trading\03_Code\vibe_replication\docs\methodology\Methodology_Working_Draft_Steps_01_to_14.docx'
$outputDirectory = 'C:\Users\housu\Documents\AI agent trading\03_Code\vibe_replication\docs\methodology\_render_steps_01_to_14'
$outputPdf = Join-Path $outputDirectory 'Methodology_Working_Draft_Steps_01_to_14.pdf'
$pagePrefix = Join-Path $outputDirectory 'page'
$pdfToPpm = 'C:\Users\housu\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$wordApplication = New-Object -ComObject Word.Application
$wordApplication.Visible = $false
$wordApplication.DisplayAlerts = 0

try {
    $wordDocument = $wordApplication.Documents.Open($inputDocx, $false, $true)
    try {
        $wordDocument.ExportAsFixedFormat($outputPdf, 17)
    }
    finally {
        $wordDocument.Close($false)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordDocument) | Out-Null
    }
}
finally {
    $wordApplication.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApplication) | Out-Null
}

& $pdfToPpm -png -r 120 $outputPdf $pagePrefix

Write-Output $outputPdf
