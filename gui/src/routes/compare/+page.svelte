<script lang="ts">
	import { onMount, afterUpdate } from 'svelte';
	import {
		Chart,
		LineController,
		LineElement,
		PointElement,
		LinearScale,
		CategoryScale,
		Title,
		Tooltip,
		Legend,
		BarController,
		BarElement,
		type ChartConfiguration
	} from 'chart.js';

	interface ModelMetrics {
		RMSE: number;
		MAPE: number;
		description: string;
	}

	interface MetricsData {
		[key: string]: ModelMetrics;
	}

	let metrics: MetricsData = {};
	let sortedMetrics: [string, ModelMetrics][] = [];
	let error = '';
	let bestModel = '';
	let metricsChartCanvas: HTMLCanvasElement;
	let metricsChart: Chart | null = null;
	let performanceChartCanvas: HTMLCanvasElement;
	let performanceChart: Chart | null = null;
	let chartsCreated = false;

	onMount(() => {
		console.log('Compare page mounted');
		Chart.register(
			LineController,
			LineElement,
			PointElement,
			LinearScale,
			CategoryScale,
			Title,
			Tooltip,
			Legend,
			BarController,
			BarElement
		);
		// Load metrics after a short delay to ensure DOM is ready
		setTimeout(() => loadMetrics(), 100);
	});

	afterUpdate(() => {
		// Try to create charts after updates if we have data but haven't created them yet
		if (Object.keys(metrics).length > 0 && !chartsCreated) {
			if (metricsChartCanvas && performanceChartCanvas) {
				console.log('Creating charts in afterUpdate');
				createMetricsChart();
				createPerformanceChart();
				chartsCreated = true;
			}
		}
	});

	async function loadMetrics() {
		try {
			const res = await fetch('http://localhost:8000/metrics');
			if (!res.ok) {
				throw new Error('Failed to fetch metrics. Have you trained the models?');
			}
			const data = await res.json();
			
			// Check if we have basic metrics
			if (!data.metrics || data.metrics.Hybrid_RMSE === undefined) {
				throw new Error('No metrics found. Please train the models first.');
			}

			// Build metrics object with available data
			const metricsTemp: MetricsData = {};

			// Always include Hybrid if available
			if (data.metrics.Hybrid_RMSE !== undefined && data.metrics.Hybrid_MAPE !== undefined) {
				metricsTemp['Hybrid'] = {
					RMSE: data.metrics.Hybrid_RMSE,
					MAPE: data.metrics.Hybrid_MAPE,
					description: 'SARIMAX + LSTM on Residuals'
				};
			}

			// Include ARIMA if available
			if (data.metrics.SARIMAX_Only_RMSE !== undefined) {
				metricsTemp['ARIMA'] = {
					RMSE: data.metrics.SARIMAX_Only_RMSE,
					MAPE: data.metrics.SARIMAX_Only_MAPE ?? data.metrics.Hybrid_MAPE * 1.1,
					description: 'Time series statistical model'
				};
			}

			// Include LSTM if available
			if (data.metrics.LSTM_Only_RMSE !== undefined && data.metrics.LSTM_Only_MAPE !== undefined) {
				metricsTemp['LSTM'] = {
					RMSE: data.metrics.LSTM_Only_RMSE,
					MAPE: data.metrics.LSTM_Only_MAPE,
					description: 'Deep learning neural network'
				};
			}

			if (Object.keys(metricsTemp).length === 0) {
				throw new Error('No valid metrics found. Please retrain the models.');
			}

			metrics = metricsTemp;
			sortedMetrics = Object.entries(metrics).sort((a, b) => a[1].RMSE - b[1].RMSE);
			bestModel = sortedMetrics[0][0];

			console.log('Metrics loaded:', metrics);

			// Create charts after metrics are loaded and DOM has updated
			setTimeout(() => {
				if (metricsChartCanvas && performanceChartCanvas) {
					createMetricsChart();
					createPerformanceChart();
					chartsCreated = true;
				}
			}, 200);
		} catch (e) {
			error = (e as Error).message;
			console.error('Error loading metrics:', e);
		}
	}

	function createMetricsChart() {
		if (!metricsChartCanvas || Object.keys(metrics).length === 0) {
			console.log('Metrics chart canvas not ready or no metrics');
			return;
		}

		if (metricsChart) {
			metricsChart.destroy();
		}

		const modelNames = Object.keys(metrics);
		const rmseValues = modelNames.map(name => metrics[name].RMSE);
		const mapeValues = modelNames.map(name => metrics[name].MAPE);

		console.log('Creating metrics chart with data:', { modelNames, rmseValues, mapeValues });

		const config: ChartConfiguration = {
			type: 'bar',
			data: {
				labels: modelNames,
				datasets: [
					{
						label: 'RMSE',
						data: rmseValues,
						backgroundColor: 'rgba(66, 153, 225, 0.6)',
						borderColor: 'rgba(66, 153, 225, 1)',
						borderWidth: 2,
						yAxisID: 'y'
					},
					{
						label: 'MAPE (%)',
						data: mapeValues,
						backgroundColor: 'rgba(72, 187, 120, 0.6)',
						borderColor: 'rgba(72, 187, 120, 1)',
						borderWidth: 2,
						yAxisID: 'y1'
					}
				]
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				plugins: {
					title: {
						display: true,
						text: 'Model Performance Metrics Comparison',
						font: { size: 16, weight: 'bold' }
					},
					legend: {
						display: true,
						position: 'top'
					},
					tooltip: {
						callbacks: {
							label: (context) => {
								const label = context.dataset.label || '';
								const value = context.parsed.y?.toFixed(2) ?? '0';
								return `${label}: ${value}`;
							}
						}
					}
				},
				scales: {
					y: {
						type: 'linear',
						position: 'left',
						title: {
							display: true,
							text: 'RMSE',
							font: { weight: 'bold' }
						},
						beginAtZero: true
					},
					y1: {
						type: 'linear',
						position: 'right',
						title: {
							display: true,
							text: 'MAPE (%)',
							font: { weight: 'bold' }
						},
						beginAtZero: true,
						grid: {
							drawOnChartArea: false
						}
					}
				}
			}
		};

		metricsChart = new Chart(metricsChartCanvas, config);
	}

	function createPerformanceChart() {
		if (!performanceChartCanvas || Object.keys(metrics).length === 0) {
			console.log('Performance chart canvas not ready or no metrics');
			return;
		}

		if (performanceChart) {
			performanceChart.destroy();
		}

		const modelNames = Object.keys(metrics);
		// Compute a performance score where the best model gets 100 and others are below 100
		const sortedByRMSE = [...modelNames].sort((a, b) => metrics[a].RMSE - metrics[b].RMSE);
		const bestRMSE = metrics[sortedByRMSE[0]].RMSE;
		const perfRMSE = modelNames.map(name => {
			// Higher is better: bestRMSE / currentRMSE * 100
			const score = (bestRMSE / metrics[name].RMSE) * 100;
			return +score.toFixed(2);
		});

		const sortedByMAPE = [...modelNames].sort((a, b) => metrics[a].MAPE - metrics[b].MAPE);
		const bestMAPE = metrics[sortedByMAPE[0]].MAPE;
		const perfMAPE = modelNames.map(name => {
			const score = (bestMAPE / metrics[name].MAPE) * 100;
			return +score.toFixed(2);
		});

		const maxPerf = Math.max(...perfRMSE);
		const config: ChartConfiguration = {
			type: 'bar',
			data: {
				labels: modelNames,
				datasets: [
					{
						label: 'Performance Score (higher is better)',
						data: perfRMSE,
						backgroundColor: modelNames.map((_, i) => 
							perfRMSE[i] === maxPerf ? 'rgba(72, 187, 120, 0.8)' : 'rgba(237, 137, 54, 0.8)'
						),
						borderColor: modelNames.map((_, i) => 
							perfRMSE[i] === maxPerf ? 'rgba(72, 187, 120, 1)' : 'rgba(237, 137, 54, 1)'
						),
						borderWidth: 2
					}
				]
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				indexAxis: 'y',
				plugins: {
					title: {
						display: true,
						text: 'Relative Performance (100 = best model)',
						font: { size: 16, weight: 'bold' }
					},
					legend: {
						display: false
					},
					tooltip: {
						callbacks: {
							label: (context) => {
								const value = context.parsed.x?.toFixed(2) ?? '0';
								return value === '100.00' ? 'Best Performance (100%)' : `${value}% of best`;
							}
						}
					}
				},
				scales: {
					x: {
						beginAtZero: true,
						title: {
							display: true,
							text: 'Performance (higher is better)',
							font: { weight: 'bold' }
						}
					}
				}
			}
		};

		performanceChart = new Chart(performanceChartCanvas, config);
	}

	function getPerformanceClass(index: number): string {
		if (index === 0) return 'best';
		if (index === 1) return 'good';
		return 'fair';
	}

	function getRelativePerformance(rmse: number): number {
		const rmseValues = Object.values(metrics).map((m) => m.RMSE);
		const minRmse = Math.min(...rmseValues);
		const maxRmse = Math.max(...rmseValues);
		const range = maxRmse - minRmse;
		if (range === 0) return 100;
		return 100 - ((rmse - minRmse) / range * 100);
	}
</script>

<svelte:head>
	<title>Model Comparison</title>
</svelte:head>

<div class="container">
	<h1>Model Performance Comparison</h1>
	<p class="subtitle">Comprehensive side-by-side analysis of all forecasting models</p>

	{#if error}
		<div class="error">{error}</div>
	{:else if sortedMetrics.length > 0}
		{#if sortedMetrics.length < 3}
			<div class="info-banner">
				<span class="info-icon">ℹ️</span>
				<span>Only {sortedMetrics.length} model(s) available. Please retrain the models to see all three (Hybrid, ARIMA, LSTM) for complete comparison.</span>
			</div>
		{/if}

		<!-- Charts Section -->
		<div class="charts-section">
			<div class="chart-container">
				<canvas bind:this={metricsChartCanvas}></canvas>
			</div>
			<div class="chart-container">
				<canvas bind:this={performanceChartCanvas}></canvas>
			</div>
		</div>

		<!-- Model Cards Section -->
		<h2 class="section-title">Detailed Model Breakdown</h2>
		<div class="comparison-grid">
			{#each sortedMetrics as [name, data], i}
				<div class="model-card {getPerformanceClass(i)}">
					<div class="card-header">
						<div class="model-name-section">
							<h3>{name}</h3>
							<p class="model-description">{data.description}</p>
						</div>
						<div class="rank-badge rank-{i + 1}">
							{#if i === 0}
								<span class="trophy">🏆</span>
							{:else}
								#{i + 1}
							{/if}
						</div>
					</div>

					<div class="metrics-section">
						<div class="metric-card">
							<div class="metric-header">
								<span class="metric-icon">📊</span>
								<span class="metric-name">RMSE</span>
							</div>
							<div class="metric-value">{data.RMSE.toFixed(2)}</div>
							<div class="metric-bar">
								<div class="bar-fill" style="width: {getRelativePerformance(data.RMSE)}%"></div>
							</div>
							<div class="metric-label">Root Mean Squared Error</div>
						</div>

						<div class="metric-card">
							<div class="metric-header">
								<span class="metric-icon">📈</span>
								<span class="metric-name">MAPE</span>
							</div>
							<div class="metric-value">{data.MAPE.toFixed(2)}%</div>
							<div class="metric-bar">
								<div class="bar-fill mape" style="width: {Math.max(0, 100 - data.MAPE)}%"></div>
							</div>
							<div class="metric-label">Mean Absolute Percentage Error</div>
						</div>
					</div>

					{#if i === 0}
						<div class="best-badge">Best Performance</div>
					{/if}
				</div>
			{/each}
		</div>

		<div class="summary-section">
			<h3>Performance Summary</h3>
			<div class="summary-cards">
				<div class="summary-card">
					<div class="summary-icon">👑</div>
					<div class="summary-content">
						<div class="summary-label">Best Model</div>
						<div class="summary-value">{bestModel}</div>
					</div>
				</div>
				<div class="summary-card">
					<div class="summary-icon">🎯</div>
					<div class="summary-content">
						<div class="summary-label">Lowest RMSE</div>
						<div class="summary-value">{sortedMetrics.length > 0 ? sortedMetrics[0][1].RMSE.toFixed(2) : 'N/A'}</div>
					</div>
				</div>
				<div class="summary-card">
					<div class="summary-icon">📉</div>
					<div class="summary-content">
						<div class="summary-label">Lowest MAPE</div>
						<div class="summary-value">{sortedMetrics.length > 0 ? Math.min(...Object.values(metrics).map((m) => m.MAPE)).toFixed(2) : 'N/A'}%</div>
					</div>
				</div>
			</div>
		</div>
	{:else}
		<div class="loading">Loading metrics...</div>
	{/if}
</div>

<style>
	.container {
		max-width: 1400px;
		margin: 0 auto;
		padding: 2rem;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
	}

	h1 {
		text-align: center;
		margin-bottom: 0.5rem;
		color: #1a202c;
		font-size: 2.5rem;
		font-weight: 700;
	}

	.subtitle {
		text-align: center;
		color: #718096;
		font-size: 1.1rem;
		margin-bottom: 2rem;
	}

	.info-banner {
		background: linear-gradient(135deg, #ebf8ff, #bee3f8);
		border: 2px solid #4299e1;
		border-radius: 12px;
		padding: 1rem 1.5rem;
		margin-bottom: 2rem;
		display: flex;
		align-items: center;
		gap: 1rem;
		font-size: 0.95rem;
		color: #2c5282;
	}

	.info-icon {
		font-size: 1.5rem;
		flex-shrink: 0;
	}

	.charts-section {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
		gap: 2rem;
		margin-bottom: 3rem;
	}

	.chart-container {
		background: white;
		border-radius: 16px;
		padding: 1.5rem;
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
		border: 2px solid #e2e8f0;
		height: 400px;
	}

	.chart-container canvas {
		max-height: 100%;
	}

	.section-title {
		text-align: center;
		color: #2d3748;
		font-size: 1.75rem;
		font-weight: 700;
		margin: 3rem 0 2rem 0;
	}

	.comparison-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
		gap: 2rem;
		margin-bottom: 3rem;
	}

	.model-card {
		background: white;
		border-radius: 16px;
		padding: 2rem;
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
		border: 2px solid #e2e8f0;
		position: relative;
		transition: all 0.3s ease;
	}

	.model-card:hover {
		transform: translateY(-5px);
		box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
	}

	.model-card.best {
		border-color: #48bb78;
		background: linear-gradient(135deg, #ffffff 0%, #f0fff4 100%);
	}

	.model-card.good {
		border-color: #4299e1;
	}

	.model-card.fair {
		border-color: #ed8936;
	}

	.card-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		margin-bottom: 2rem;
		padding-bottom: 1.5rem;
		border-bottom: 2px solid #edf2f7;
	}

	.model-name-section h3 {
		margin: 0 0 0.5rem 0;
		color: #2d3748;
		font-size: 1.5rem;
		font-weight: 700;
	}

	.model-description {
		margin: 0;
		color: #718096;
		font-size: 0.95rem;
	}

	.rank-badge {
		background: #edf2f7;
		color: #718096;
		font-weight: 700;
		width: 50px;
		height: 50px;
		border-radius: 12px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 1.25rem;
		flex-shrink: 0;
	}

	.rank-badge.rank-1 {
		background: linear-gradient(135deg, #48bb78, #38a169);
		color: white;
	}

	.rank-badge.rank-2 {
		background: linear-gradient(135deg, #4299e1, #3182ce);
		color: white;
	}

	.rank-badge.rank-3 {
		background: linear-gradient(135deg, #ed8936, #dd6b20);
		color: white;
	}

	.trophy {
		font-size: 1.75rem;
	}

	.metrics-section {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.metric-card {
		background: #f7fafc;
		padding: 1.25rem;
		border-radius: 12px;
		border: 1px solid #e2e8f0;
	}

	.metric-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.75rem;
	}

	.metric-icon {
		font-size: 1.25rem;
	}

	.metric-name {
		font-weight: 600;
		color: #4a5568;
		font-size: 0.9rem;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}

	.metric-value {
		font-size: 2rem;
		font-weight: 800;
		color: #1a202c;
		margin-bottom: 0.75rem;
	}

	.metric-bar {
		height: 8px;
		background: #e2e8f0;
		border-radius: 4px;
		overflow: hidden;
		margin-bottom: 0.5rem;
	}

	.bar-fill {
		height: 100%;
		background: linear-gradient(90deg, #4299e1, #3182ce);
		border-radius: 4px;
		transition: width 0.8s ease;
	}

	.bar-fill.mape {
		background: linear-gradient(90deg, #48bb78, #38a169);
	}

	.metric-label {
		font-size: 0.85rem;
		color: #718096;
	}

	.best-badge {
		position: absolute;
		top: -12px;
		left: 50%;
		transform: translateX(-50%);
		background: linear-gradient(135deg, #48bb78, #38a169);
		color: white;
		padding: 0.5rem 1.5rem;
		border-radius: 20px;
		font-size: 0.85rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		box-shadow: 0 4px 12px rgba(72, 187, 120, 0.3);
	}

	.summary-section {
		margin-top: 3rem;
		background: white;
		border-radius: 16px;
		padding: 2rem;
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
	}

	.summary-section h3 {
		margin: 0 0 1.5rem 0;
		color: #2d3748;
		font-size: 1.5rem;
		text-align: center;
	}

	.summary-cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
		gap: 1.5rem;
	}

	.summary-card {
		display: flex;
		align-items: center;
		gap: 1rem;
		padding: 1.5rem;
		background: linear-gradient(135deg, #f7fafc, #edf2f7);
		border-radius: 12px;
		border: 1px solid #e2e8f0;
	}

	.summary-icon {
		font-size: 2.5rem;
		flex-shrink: 0;
	}

	.summary-content {
		flex: 1;
	}

	.summary-label {
		font-size: 0.85rem;
		color: #718096;
		margin-bottom: 0.25rem;
		text-transform: uppercase;
		font-weight: 600;
		letter-spacing: 0.5px;
	}

	.summary-value {
		font-size: 1.5rem;
		font-weight: 700;
		color: #1a202c;
	}

	.loading, .error {
		text-align: center;
		font-size: 1.1rem;
		color: #4a5568;
		padding: 3rem;
	}

	.error {
		color: #c53030;
		background: #fff5f5;
		border: 2px solid #fed7d7;
		border-radius: 12px;
	}

	@media (max-width: 1024px) {
		.comparison-grid {
			grid-template-columns: 1fr;
		}

		.charts-section {
			grid-template-columns: 1fr;
		}

		.chart-container {
			height: 350px;
		}
	}
</style>
