import React, { useState, useEffect, useRef } from 'react'
import './styles.css'

function formatNumber(v){
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return Number(v).toFixed(3)
}

export default function InterventionSlider({apiUrl = 'http://127.0.0.1:8000', interventionVariable='x', targetVariable='y'}){
  const [value, setValue] = useState(0)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const timeoutRef = useRef(null)

  useEffect(()=>{
    // Debounce requests
    if(timeoutRef.current) clearTimeout(timeoutRef.current)
    timeoutRef.current = setTimeout(()=>{
      runIntervention(value)
    }, 300)
    return ()=> clearTimeout(timeoutRef.current)
  }, [value])

  async function runIntervention(val){
    setLoading(true)
    try{
      const resp = await fetch(`${apiUrl}/api/v1/causal/intervene`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ intervention_variable: interventionVariable, intervention_value: parseFloat(val), target_variable: targetVariable })
      })
      if(!resp.ok){
        const txt = await resp.text()
        setResult({error: txt})
      }else{
        const data = await resp.json()
        setResult(data)
      }
    }catch(err){
      setResult({error: err.message})
    }finally{
      setLoading(false)
    }
  }

  return (
    <section className="card" aria-labelledby="intervention-heading">
      <div style={{marginBottom:12}}>
        <label id="intervention-heading" style={{display:'block', marginBottom:6}}>Intervention: <strong>{interventionVariable}</strong></label>

        <div style={{display:'flex', gap:8, alignItems:'center'}}>
          <input
            aria-label={`Intervention value for ${interventionVariable}`}
            type="range"
            min={-10}
            max={10}
            value={value}
            onChange={e=>setValue(Number(e.target.value))}
            step={0.1}
            className="range"
          />

          <div style={{display:'flex', flexDirection:'column', alignItems:'flex-end'}}>
            <input
              aria-label={`Numeric input for ${interventionVariable}`}
              value={value}
              onChange={e=>setValue(Number(e.target.value))}
              style={{width:80, padding:6, borderRadius:6, border:'1px solid #e6e9ee'}}
              type="number"
              step="0.1"
            />
            <div style={{fontSize:12, marginTop:6, color:'#666'}}>{loading ? 'Computing...' : 'Ready'}</div>
          </div>
        </div>
      </div>

      <div role="status" aria-live="polite" className="card" style={{background:'#fbfdff'}}>
        {result && result.error ? (
          <div className="error">Error: {result.error}</div>
        ) : result ? (
          <article>
            <div style={{fontSize:18, color:'#111'}}><strong>Predicted {targetVariable}: {formatNumber(result.predicted_effect)}</strong></div>
            <div style={{color:'#333'}}>Baseline: {formatNumber(result.baseline_value)} &nbsp; | &nbsp; Change: {isNaN(result.change_percentage) ? '—' : result.change_percentage.toFixed(1)+'%'}</div>
            <div style={{marginTop:8}}>
              <div>Std: {formatNumber(result.predicted_std)} </div>
              <div>95% CI: {formatNumber(result.ci_lower)} — {formatNumber(result.ci_upper)}</div>
            </div>
          </article>
        ) : (
          <div>Move the slider to run intervention.</div>
        )}
      </div>
    </section>
  )
}
