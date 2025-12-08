import React from 'react'
import InterventionSlider from './InterventionSlider'

export default {
  title: 'Intervention/Slider',
  component: InterventionSlider,
}

const Template = (args) => <div style={{padding:20, maxWidth:600}}><InterventionSlider {...args} /></div>

export const Default = Template.bind({})
Default.args = {
  apiUrl: 'http://127.0.0.1:8000',
  interventionVariable: 'CH_DISASTER_count',
  targetVariable: 'CH_INFRASTRUCTURE_DAMAGE_count'
}
