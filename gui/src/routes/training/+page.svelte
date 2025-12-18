<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Chart,
		BarController,
		BarElement,
		CategoryScale,
		LinearScale,
		Title,
		Tooltip,
		Legend,
		type ChartConfiguration
	} from 'chart.js';

	// Register Chart.js components
	Chart.register(BarController, BarElement, CategoryScale, LinearScale, Title, Tooltip, Legend);

	let trainFile: File | null = null;
	let testFile: File | null = null;
	let seed = 42;
	let loading = false;
	let metrics: any = null;
	let error = '';
	let chartCanvas: HTMLCanvasElement;
	let chart: Chart | null = null;

	const API_BASE = 'http://localhost:8000';

	function handleTrainFileChange(e: Event) {
		const target = e.target as HTMLInputElement;
		if (target.files && target.files[0]) {
			trainFile = target.files[0];
		}
	}

	function handleTestFileChange(e: Event) {
		const target = e.target as HTMLInputElement;
		if (target.files && target.files[0]) {
			testFile = target.files[0];
		}
	}

	async function handleTrain() {
		if (!trainFile || !testFile) {
			error = 'Please upload both train and test CSV files';
			return;
		}

		loading = true;
		error = '';
		metrics = null;

		try {
			const formData = new FormData();
			formData.append('train_file', trainFile);
			formData.append('test_file', testFile);
			formData.append('seed', seed.toString());

			const response = await fetch(`${API_BASE}/train`, {
				method: 'POST',
				body: formData
			});

			if (!response.ok) {
				const errData = await response.json();
				throw new Error(errData.detail || 'Training failed');
			}

			const data = await response.json();
			metrics = data.metrics;
		} catch (err: any) {
			error = err.message || 'An error occurred during training';
			console.error('Training error:', err);
		} finally {
			loading = false;
		}
	}

	async function downloadModel() {
		try {
			const response = await fetch(`${API_BASE}/model/download`);
			if (!response.ok) throw new Error('Failed to download model');

			const blob = await response.blob();
			const url = window.URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = 'lstm_model.keras';
			document.body.appendChild(a);
			a.click();
			window.URL.revokeObjectURL(url);
			document.body.removeChild(a);
		} catch (err: any) {
			error = err.message || 'Failed to download model';
		}
	}

	function renderComparisonChart() {
		if (!metrics || !chartCanvas) return;

		// Destroy existing chart
		if (chart) {
			chart.destroy();
		}

		const config: ChartConfiguration = {
			type: 'bar',
			data: {
				labels: ['RMSE', 'MAPE (%)'],
				datasets: [
					{
						label: 'ARIMA',
						data: [metrics.SARIMAX_RMSE, metrics.SARIMAX_MAPE],
						backgroundColor: 'rgba(74, 144, 226, 0.7)',
						borderColor: '#4a90e2',
						borderWidth: 2
					},
					{
						label: 'LSTM',
						data: [metrics.LSTM_RMSE, metrics.LSTM_MAPE],
						backgroundColor: 'rgba(249, 115, 22, 0.7)',
						borderColor: '#f97316',
						borderWidth: 2
					}
				]
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				plugins: {
					title: {
						display: true,
						text: 'Model Performance Comparison',
						font: {
							size: 18,
							weight: 'bold'
						}
					},
					legend: {
						display: true,
						position: 'top'
					},
					tooltip: {
						callbacks: {
							label: function (context) {
								let label = context.dataset.label || '';
								if (label) {
									label += ': ';
								}
								if (context.parsed.y !== null) {
									label += context.parsed.y.toFixed(2);
									if (context.dataIndex === 1) {
										label += '%';
									}
								}
								return label;
							}
						}
					}
				},
				scales: {
					y: {
						beginAtZero: true,
						title: {
							display: true,
							text: 'Error Value',
							font: {
								size: 14,
								weight: 'bold'
							}
						}
					}
				}
			}
		};

		chart = new Chart(chartCanvas, config);
	}

	// Re-render chart when metrics change
	$: if (metrics && chartCanvas) {
		renderComparisonChart();
	}
</script>

<svelte:head>
	<title>Train Models - E-commerce Demand Forecasting</title>
</svelte:head>

<div class="container">
	<h1>Train ARIMA & LSTM Models</h1>
	<p class="subtitle">Upload your training and test datasets to train forecasting models</p>

	<div class="upload-section">
		<div class="file-input-group">
			<label for="train-file">
				<span class="label-text">Training Data (CSV)</span>
				<input
					id="train-file"
					type="file"
					accept=".csv"
					on:change={handleTrainFileChange}
					disabled={loading}
				/>
				{#if trainFile}
					<span class="file-name">✓ {trainFile.name}</span>
				{/if}
			</label>
		</div>

		<div class="file-input-group">
			<label for="test-file">
				<span class="label-text">Test Data (CSV)</span>
				<input
					id="test-file"
					type="file"
					accept=".csv"
					on:change={handleTestFileChange}
					disabled={loading}
				/>
				{#if testFile}
					<span class="file-name">✓ {testFile.name}</span>
				{/if}
			</label>
		</div>
	</div>

	<div class="options">
		<label class="option-row">
			<span>Random Seed:</span>
			<input type="number" bind:value={seed} disabled={loading} />
		</label>


	</div>

	<button class="train-btn" on:click={handleTrain} disabled={loading || !trainFile || !testFile}>
		{#if loading}
			<span class="spinner"></span>
			Training Models...
		{:else}
			Train Models
		{/if}
	</button>

	{#if error}
		<div class="error-box">
			<strong>Error:</strong> {error}
		</div>
	{/if}

	{#if metrics}
		<div class="results">
			<h2>Training Results</h2>

			<!-- Interactive Comparison Chart -->
			<div class="chart-container">
				<canvas bind:this={chartCanvas}></canvas>
			</div>

			<div class="metrics-grid">
				<div class="metric-card sarimax">
					<h3>ARIMA</h3>
					<div class="metric-row">
						<span>RMSE:</span>
						<span class="value">{metrics.SARIMAX_RMSE.toFixed(2)}</span>
					</div>
					<div class="metric-row">
						<span>MAPE:</span>
						<span class="value">{metrics.SARIMAX_MAPE.toFixed(2)}%</span>
					</div>
				</div>

				<div class="metric-card lstm">
					<h3>LSTM</h3>
					<div class="metric-row">
						<span>RMSE:</span>
						<span class="value">{metrics.LSTM_RMSE.toFixed(2)}</span>
					</div>
					<div class="metric-row">
						<span>MAPE:</span>
						<span class="value">{metrics.LSTM_MAPE.toFixed(2)}%</span>
					</div>
				</div>
			</div>

			<div class="comparison">
				{#if metrics.LSTM_RMSE < metrics.SARIMAX_RMSE}
					<p class="winner">🎉 LSTM outperforms ARIMA!</p>
				{:else}
					<p class="info">ARIMA performed better on this dataset.</p>
				{/if}
			</div>

			<button class="download-btn" on:click={downloadModel}>
				Download LSTM Model (.keras)
			</button>
		</div>
	{/if}
</div>

<style>
	.container {
		max-width: 900px;
		margin: 0 auto;
		padding: 2rem;
	}

	h1 {
		font-size: 2rem;
		margin-bottom: 0.5rem;
		color: #1a1a1a;
	}

	.subtitle {
		color: #666;
		margin-bottom: 2rem;
	}

	.upload-section {
		display: grid;
		gap: 1.5rem;
		margin-bottom: 2rem;
	}

	.file-input-group {
		border: 2px dashed #ddd;
		border-radius: 8px;
		padding: 1.5rem;
		transition: border-color 0.3s;
	}

	.file-input-group:hover {
		border-color: #4a90e2;
	}

	.label-text {
		display: block;
		font-weight: 600;
		margin-bottom: 0.5rem;
		color: #333;
	}

	input[type='file'] {
		width: 100%;
		padding: 0.5rem;
		font-size: 0.95rem;
	}

	.file-name {
		display: block;
		margin-top: 0.5rem;
		color: #22c55e;
		font-weight: 500;
	}

	.options {
		background: #f8f9fa;
		padding: 1.5rem;
		border-radius: 8px;
		margin-bottom: 2rem;
	}

	.option-row {
		display: flex;
		align-items: center;
		gap: 1rem;
		margin-bottom: 1rem;
	}

	.option-row:last-child {
		margin-bottom: 0;
	}

	.option-row input[type='number'] {
		width: 100px;
		padding: 0.4rem;
		border: 1px solid #ddd;
		border-radius: 4px;
	}

	.train-btn {
		width: 100%;
		padding: 1rem;
		background: #4a90e2;
		color: white;
		border: none;
		border-radius: 8px;
		font-size: 1.1rem;
		font-weight: 600;
		cursor: pointer;
		transition: background 0.3s;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.5rem;
	}

	.train-btn:hover:not(:disabled) {
		background: #3a7bc8;
	}

	.train-btn:disabled {
		background: #ccc;
		cursor: not-allowed;
	}

	.spinner {
		border: 3px solid rgba(255, 255, 255, 0.3);
		border-top-color: white;
		border-radius: 50%;
		width: 20px;
		height: 20px;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	.error-box {
		margin-top: 1.5rem;
		padding: 1rem;
		background: #fee;
		border: 1px solid #fcc;
		border-radius: 8px;
		color: #c00;
	}

	.results {
		margin-top: 2rem;
		padding: 2rem;
		background: #f8f9fa;
		border-radius: 12px;
	}

	.results h2 {
		margin-bottom: 1.5rem;
		color: #1a1a1a;
	}

	.metrics-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
		gap: 1.5rem;
		margin-bottom: 1.5rem;
	}

	.metric-card {
		background: white;
		padding: 1.5rem;
		border-radius: 8px;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}

	.metric-card h3 {
		margin-bottom: 1rem;
		font-size: 1.2rem;
	}

	.metric-card.sarimax h3 {
		color: #4a90e2;
	}

	.metric-card.lstm h3 {
		color: #f97316;
	}

	.metric-row {
		display: flex;
		justify-content: space-between;
		margin-bottom: 0.5rem;
		font-size: 1rem;
	}

	.metric-row .value {
		font-weight: 700;
		font-size: 1.1rem;
	}

	.comparison {
		text-align: center;
		margin: 1.5rem 0;
	}

	.winner {
		color: #22c55e;
		font-weight: 700;
		font-size: 1.1rem;
	}

	.info {
		color: #666;
	}

	.download-btn {
		width: 100%;
		padding: 0.8rem;
		background: #22c55e;
		color: white;
		border: none;
		border-radius: 8px;
		font-weight: 600;
		cursor: pointer;
		transition: background 0.3s;
	}

	.download-btn:hover {
		background: #16a34a;
	}

	.chart-container {
		background: white;
		padding: 1.5rem;
		border-radius: 8px;
		margin-bottom: 1.5rem;
		height: 350px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
	}

	.chart-container canvas {
		max-height: 100%;
	}
</style>
