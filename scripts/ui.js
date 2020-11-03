// IRIS_CLASSES, need to import classList and its name
function logitsToSpans(logits) {
    let idMax = -1;
    let maxLogit = Number.NEGATIVE_INFINITY;
    for(let i = 0; i < logits.length; ++i){
        if(logits[i] < maxLogit){
            maxLogit = logits[i];
            idMax = i;
        }
    }
    const span = []
    for(let i = 0; i < logits.length; ++i){
        const logitSpan = document.createElement('span');
        logitSpan.textContent = logits[i].toFixed(3);
        if (i == idMax){
            logitSpan.style['font-weight'] = 'bold';
            if(logit[i] < CLASS_TRESH){
                logitSpan.classList = ['wrong-prediction'];
            }
        }
        logitSpan.classList = ['logit-span'];
        span.push(logitSpan)
    }
    return span

}
export function renderEvaluateTable(xData , yPred, logits) {
    const tableBody = document.getElementById('evaluate-tbody');
// yTrue will be given outside,hardcode here, the length of yTrue(items to show)
// yTrue will be 0, if the prediction below the thredhold
    const yTrue = [1]
    for (let i = 0; i < yTrue.length; ++i){
        const row = document.createElement('tr');
        for (let j = 0; j < 2; ++j) {
            const cell = document.createElement('td')
            cell.textContent = xData[2*i + j];
            console.log(xData)
            row.appendChild(cell)
        }
        const predCell = document.createElement('td');
        predCell.textContent = yPred[i];
        row.appendChild(predCell);

        // const idCell = yPred[i];
        // row.appendChild(idCell);

        const logitsCell = document.createElement('td');
        const logitSpan = document.createElement('span');
        console.log(logits)
        logitSpan.textContent = logits
        if (logits < 0.5){
            logitSpan.classList = ["wrong-prediction"]
        }
        else{
            logitSpan.classList = ['logit-span']
        }
        // const exampleLogits = 
        //     logits.slice(i*IRIS_NUM_CLASSES, (i+1)*IRIS_NUM_CLASSES);
        // logitsToSpans(exampleLogits).map(logitSpan =>{
        //     logitsCell.appendChild(logitSpan);
        // })
        logitsCell.appendChild(logitSpan)
        row.appendChild(logitsCell)
        tableBody.appendChild(row)
    }
}
export function clearEvaluateTable() {
    const tableBody = document.getElementById('evaluate-tbody');
    while (tableBody.children.length > 1) {
    tableBody.removeChild(tableBody.children[1]);
    }
}