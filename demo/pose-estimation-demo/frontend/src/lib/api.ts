export async function runExperiment(steps=10){const r=await fetch('/experiments/run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({steps})});return r.json()}
