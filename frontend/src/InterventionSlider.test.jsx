import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'
import InterventionSlider from './InterventionSlider'

// Mock fetch globally to avoid network calls in Storybook tests
beforeAll(() => {
  global.fetch = jest.fn(() => Promise.resolve({
    ok: true,
    json: () => Promise.resolve({
      predicted_effect: 1.23,
      baseline_value: 0.5,
      change_percentage: 146,
      predicted_std: 0.12,
      ci_lower: 0.98,
      ci_upper: 1.48
    })
  }))
})

afterAll(() => {
  delete global.fetch
})

test('renders slider and displays prediction after change', async () => {
  render(<div style={{padding:20, maxWidth:600}}><InterventionSlider apiUrl="http://127.0.0.1:8000" interventionVariable="X" targetVariable="Y" /></div>)

  // Slider should be present
  const slider = screen.getByLabelText(/Intervention value for X/i)
  expect(slider).toBeInTheDocument()

  // Simulate value change by setting value attribute and triggering input event
  slider.value = 2
  slider.dispatchEvent(new Event('input', { bubbles: true }))

  // Wait for mocked fetch to resolve and predicted value to appear
  await waitFor(() => expect(screen.getByText(/Predicted Y/i)).toBeInTheDocument())
  expect(screen.getByText(/1.230/)).toBeInTheDocument()
})
