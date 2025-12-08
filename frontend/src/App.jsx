import React from 'react'
import InterventionSlider from './InterventionSlider'

export default function App(){
  return (
    <div style={{fontFamily:'Arial, sans-serif', padding:20, maxWidth:720}}>
      <h2>ResilientFlow — Intervention Demo</h2>
      <p>Move the slider to set an intervention on a variable and see predicted effect and 95% CI.</p>
      <InterventionSlider
        apiUrl={process.env.VITE_API_URL || 'http://127.0.0.1:8000'}
        interventionVariable={'CH_DISASTER_count'}
        targetVariable={'CH_INFRASTRUCTURE_DAMAGE_count'}
      />
    </div>
  )
}
