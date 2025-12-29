<script lang="ts">
	import { onMount } from 'svelte';
	import ExplanationCard from '$lib/ExplanationCard.svelte';
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
		Filler,
		BarController,
		BarElement,
		type ChartConfiguration,
		Decimation
	} from 'chart.js';

	onMount(() => {
		Chart.register(
			LineController,
			LineElement,
			PointElement,
			LinearScale,
			CategoryScale,
			Title,
			Tooltip,
			Legend,
			Filler,
			BarController,
			BarElement,
			Decimation
		);
	});

	let dataFile: File | null = null;
	let quarters = 2;
	let loading = false;
	let forecast: any = null;
	let error = '';
	let chartCanvas: HTMLCanvasElement;
	let quarterlyChartCanvas: HTMLCanvasElement;
	let weekdayCanvas: HTMLCanvasElement;
	let monthlyCanvas: HTMLCanvasElement;
	let skuComparisonCanvas: HTMLCanvasElement;
	let distributionCanvas: HTMLCanvasElement;
	let chart: Chart | null = null;
	let quarterlyChart: Chart | null = null;
	let weekdayChart: Chart | null = null;
	let monthlyChart: Chart | null = null;
	let skuComparisonChart: Chart | null = null;
	let distributionChart: Chart | null = null;
	let selectedSku: string = 'all';
	let selectedExplanation: any = null;
	let selectedModel: string = 'hybrid';

	$: {
		if (forecast && selectedSku && forecast.forecasts[selectedSku]) {
			selectedExplanation = forecast.forecasts[selectedSku].explanation;
		} else {
			selectedExplanation = null;
		}
	}

	const API_BASE = import.meta.env.VITE_API_BASE || '';

	// Downsample data for charts to prevent browser freezing
	function downsampleForChart(data: any[], maxPoints: number = 400): any[] {
		if (!data || data.length <= maxPoints) return data;
		
		const step = data.length / maxPoints;
		const result: any[] = [];
		
		for (let i = 0; i < maxPoints; i++) {
			const startIdx = Math.floor(i * step);
			const endIdx = Math.floor((i + 1) * step);
			const chunk = data.slice(startIdx, Math.min(endIdx, data.length));
			
			if (chunk.length > 0) {
				const midIdx = Math.floor(chunk.length / 2);
				const point = { ...chunk[midIdx] };
				
				// Average numerical values for smoother visualization
				if (chunk.length > 1) {
					const avgUnits = chunk.reduce((sum, c) => sum + (c.predicted_units || 0), 0) / chunk.length;
					point.predicted_units = Math.round(avgUnits);
				}
				result.push(point);
			}
		}
		
		return result;
	}

	function handleFileChange(e: Event) {
		const target = e.target as HTMLInputElement;
		if (target.files && target.files[0]) {
			dataFile = target.files[0];
		}
	}

	async function handlePredict() {
		if (!dataFile) {
			error = 'Please upload a CSV file';
			return;
		}

		loading = true;
		error = '';
		forecast = null;

		try {
			const formData = new FormData();
			formData.append('data_file', dataFile);
			formData.append('quarters', quarters.toString());
			// Backend now defaults to hybrid, but we send it for legacy compat if needed or just omit
			formData.append('model_type', 'hybrid');

const response = await fetch(`${API_BASE}/api/forecast_future`, {
				method: 'POST',
				body: formData
			});

			if (!response.ok) {
				const errData = await response.json();
				throw new Error(errData.detail || 'Forecast failed');
			}

			const data = await response.json();
		console.log('Received forecast data:', data);
		console.log('total_daily sample:', data.total_daily ? data.total_daily.slice(0,8) : null);
		forecast = data;
		selectedSku = data.skus[0];
		console.log('Set selectedSku to:', selectedSku);
		} catch (err: any) {
			error = err.message || 'An error occurred during forecasting';
			console.error('Forecast error:', err);
		} finally {
			loading = false;
		}
	}

	function downloadPredictions() {
		if (!forecast) return;

		const currentData = selectedSku === 'all' 
			? forecast.total_daily 
			: forecast.forecasts[selectedSku].daily;

		const csv = [
			['Date', 'Predicted Units', 'SKU'],
			...currentData.map((item: any) => [
				item.date, 
				item.predicted_units.toFixed(0),
				selectedSku
			])
		]
			.map((row) => row.join(','))
			.join('\n');

		const blob = new Blob([csv], { type: 'text/csv' });
		const url = window.URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `forecast_${selectedSku}_${quarters}q.csv`;
		document.body.appendChild(a);
		a.click();
		window.URL.revokeObjectURL(url);
		document.body.removeChild(a);
	}

	function calculateMovingAverage(data: number[], window: number = 7): (number | null)[] {
		const result: (number | null)[] = [];
		for (let i = 0; i < data.length; i++) {
			if (i < window - 1) {
				result.push(null);
			} else {
				const sum = data.slice(i - window + 1, i + 1).reduce((a, b) => a + b, 0);
				result.push(sum / window);
			}
		}
		return result;
	}

	function renderChart() {
		if (!forecast || !chartCanvas) {
			console.log('renderChart skipped:', { forecast: !!forecast, chartCanvas: !!chartCanvas });
			return;
		}

		console.log('renderChart called for SKU:', selectedSku);

		const rawData = selectedSku === 'all' 
			? forecast.total_daily 
			: forecast.forecasts[selectedSku].daily;

		// Data is already downsampled by backend, but apply client-side limit as safety
		const currentData = downsampleForChart(rawData, 500);

		console.log('currentData length:', currentData?.length, '(raw:', rawData?.length, ')');

		if (chart) {
			chart.destroy();
		}

		const dates = currentData.map((item: any) => item.date);
		const values = currentData.map((item: any) => item.predicted_units);

		const ma7 = calculateMovingAverage(values, 7);
		const ma14 = calculateMovingAverage(values, 14);

		console.log('MA samples (client)', { values: values.slice(0,6), ma7: ma7.slice(0,6), ma14: ma14.slice(0,6) });

		const config: ChartConfiguration = {
			type: 'line',
			data: {
				labels: dates,
				datasets: [
					{
						label: '7-Day Moving Average',
						data: ma7,
						borderColor: '#3b82f6',
						backgroundColor: 'rgba(59, 130, 246, 0.06)',
						borderWidth: 2.5,
						pointRadius: 0,
						fill: false,
						tension: 0.3,
						borderDash: [5, 5]
					},
					{
						label: selectedSku === 'all' ? '14-Day MA (Total)' : `14-Day MA - ${selectedSku}`,
						data: ma14,
						borderColor: '#f97316',
						backgroundColor: 'rgba(249, 115, 22, 0.12)',
						borderWidth: 3,
						pointRadius: 0,
						fill: false,
						cubicInterpolationMode: 'monotone',
						tension: 0.45
					}
				]
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				animation: false, // Disable animation for performance
				interaction: {
					mode: 'index',
					intersect: false
				},
				plugins: {
					decimation: {
						enabled: true,
						algorithm: 'lttb',
						samples: 300
					},
					title: {
						display: true,
						text: `Future Demand Forecast - ${forecast.forecast_horizon} (${forecast.model})`,
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
									label += context.parsed.y.toFixed(1) + ' units';
								}
								return label;
							}
						}
					}
				},
				scales: {
					x: {
						display: true,
						title: {
							display: true,
							text: 'Date',
							font: {
								size: 14,
								weight: 'bold'
							}
						},
						ticks: {
							maxRotation: 45,
							minRotation: 45,
							maxTicksLimit: 15
						}
					},
					y: {
						display: true,
						title: {
								display: true,
								text: 'Predicted Units',
								font: {
									size: 14,
									weight: 'bold'
								}
							},
						beginAtZero: false
					}
				}
			}
		};

		chart = new Chart(chartCanvas, config);

		try {
			if (chart && chart.data && chart.data.datasets && chart.data.datasets.length > 0) {
				chart.data.datasets.forEach((d: any, idx: number) => {
					if (idx === 0) {
						d.pointRadius = 0; d.tension = 0.2; d.cubicInterpolationMode = 'monotone';
					} else {
						d.pointRadius = 0; d.tension = d.tension ?? 0.35; d.cubicInterpolationMode = d.cubicInterpolationMode ?? 'monotone';
					}
				});
				const hasMA14 = chart.data.datasets.some((ds: any) => ds.label && ds.label.toString().includes('14-Day'));
				if (!hasMA14) {
					const ma14 = currentData.map((item: any) => item.ma14 !== undefined ? item.ma14 : null);
					chart.data.datasets.push({
						label: selectedSku === 'all' ? '14-Day MA (Total)' : `14-Day MA - ${selectedSku}`,
						data: ma14,
						borderColor: '#f97316',
						borderWidth: 3,
						pointRadius: 0,
						fill: false,
						cubicInterpolationMode: 'monotone',
						tension: 0.45
					});
				}

				chart.update();
				adjustChartYAxis(chart, values, ma7, ma14);
			}
		} catch (err) {
			console.warn('Post-chart adjustments failed:', err);
		}
	}

	let _yAdjustInProgress = false;
function adjustChartYAxis(chart:any, values:any[], ma7:any[], ma14:any[]) {
	// Prevent re-entrancy
	if (!chart || _yAdjustInProgress) return;
	_yAdjustInProgress = true;

	try {
		const numeric = [
			...values,
			...ma7.filter((v:any)=>v!==null),
			...ma14.filter((v:any)=>v!==null)
		].filter((v:any)=>v!==null && !isNaN(v));

		if (!numeric.length) {
			_yAdjustInProgress = false;
			return;
		}
		const mn = Math.min(...numeric);
		const mx = Math.max(...numeric);
		
		requestAnimationFrame(() => {
			try {
				if (!chart.options) chart.options = {};
				if (!chart.options.scales) chart.options.scales = {};
				if (!chart.options.scales.y) chart.options.scales.y = {};
				
				const range = mx - mn;

				if (range === 0) {
					const y = mx;
					chart.options.scales.y.min = Math.max(0, y - 1);
					chart.options.scales.y.max = y + 1;
				} else {
					const pad = range * 0.15; // 15% padding
					chart.options.scales.y.min = Math.max(0, mn - pad);
					chart.options.scales.y.max = mx + pad;
				}
				chart.update();
			} catch (err) {
				console.warn('adjustChartYAxis inner failed', err);
			} finally {
				_yAdjustInProgress = false;
			}
		});
	} catch (e) { 
		_yAdjustInProgress = false; 
		console.warn('adjustChartYAxis failed', e); 
	}
}


	function renderQuarterlyChart() {
		if (!forecast || !quarterlyChartCanvas || selectedSku === 'all') {
			console.log('renderQuarterlyChart skipped:', { 
				forecast: !!forecast, 
				quarterlyChartCanvas: !!quarterlyChartCanvas, 
				selectedSku 
			});
			return;
		}

		console.log('renderQuarterlyChart called for SKU:', selectedSku);

		const quarterlyData = forecast.forecasts[selectedSku].quarterly;
		console.log('quarterlyData:', quarterlyData);

		if (quarterlyChart) {
			quarterlyChart.destroy();
		}

		const config: ChartConfiguration = {
			type: 'bar',
			data: {
				labels: quarterlyData.map((q: any) => q.quarter_label),
				datasets: [
					{
						label: 'Predicted Demand',
						data: quarterlyData.map((q: any) => q.predicted_units),
						backgroundColor: 'rgba(34, 197, 94, 0.7)',
						borderColor: '#22c55e',
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
						text: `Quarterly Forecast - ${selectedSku}`,
						font: {
							size: 16,
							weight: 'bold'
						}
					},
					legend: {
						display: false
					},
					tooltip: {
						callbacks: {
							label: function (context) {
								const y = context.parsed?.y ?? 0;
								return `${y.toFixed(0)} units`;
							}
						}
					}
				},
				scales: {
					y: {
						beginAtZero: true,
						title: {
							display: true,
							text: 'Total Units',
							font: {
								size: 14,
								weight: 'bold'
							}
						}
					}
				}
			}
		};

		quarterlyChart = new Chart(quarterlyChartCanvas, config);
	// Adjust y-axis so small differences in quarterly bars are visible
	adjustChartYAxis(quarterlyChart, quarterlyData.map((q:any)=>q.predicted_units), [], []);
	}

	function renderWeekdayChart() {
		if (!forecast || !weekdayCanvas || selectedSku === 'all') return;

		if (weekdayChart) weekdayChart.destroy();

		// Use full data for aggregation (already downsampled from backend)
		const dailyData = forecast.forecasts[selectedSku].daily;
		const weekdayMap: any = { 0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun' };
		const weekdayAgg: any = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] };

		dailyData.forEach((d: any) => {
			const dow = new Date(d.date).getDay();
			const adjustedDow = dow === 0 ? 6 : dow - 1; // Convert Sunday=0 to Monday=0
			weekdayAgg[adjustedDow].push(d.predicted_units);
		});

		const weekdayAvgs = Object.keys(weekdayAgg).map(k => {
			const vals = weekdayAgg[k];
			return vals.length ? vals.reduce((a: number, b: number) => a + b, 0) / vals.length : 0;
		});

		const cfg: ChartConfiguration = {
			type: 'bar',
			data: {
				labels: Object.values(weekdayMap),
				datasets: [{
					label: 'Avg Forecast by Day',
					data: weekdayAvgs,
					backgroundColor: 'rgba(99, 102, 241, 0.8)',
					borderColor: '#6366f1',
					borderWidth: 2
				}]
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				animation: false,
				plugins: {
					title: { display: true, text: `Day-of-Week Pattern - ${selectedSku}`, font: { size: 16, weight: 'bold' } },
					legend: { display: false }
				},
				scales: { y: { beginAtZero: true, title: { display: true, text: 'Avg Units' } } }
			}
		};

		weekdayChart = new Chart(weekdayCanvas, cfg);
	}

	function renderMonthlyChart() {
		if (!forecast || !monthlyCanvas || selectedSku === 'all') return;

		if (monthlyChart) monthlyChart.destroy();

		// Use data from backend (already downsampled)
		const dailyData = forecast.forecasts[selectedSku].daily;
		const monthlyAgg: any = {};

		dailyData.forEach((d: any) => {
			const dt = new Date(d.date);
			const monthKey = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, '0')}`;
			if (!monthlyAgg[monthKey]) monthlyAgg[monthKey] = 0;
			monthlyAgg[monthKey] += d.predicted_units;
		});

		const labels = Object.keys(monthlyAgg).sort();
		const values = labels.map(k => monthlyAgg[k]);

		const cfg: ChartConfiguration = {
			type: 'bar',
			data: {
				labels,
				datasets: [{
					label: 'Monthly Total',
					data: values,
					backgroundColor: 'rgba(16, 185, 129, 0.8)',
					borderColor: '#10b981',
					borderWidth: 2
				}]
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				animation: false,
				plugins: {
					title: { display: true, text: `Monthly Forecast - ${selectedSku}`, font: { size: 16, weight: 'bold' } },
					legend: { display: false }
				},
				scales: { y: { beginAtZero: true, title: { display: true, text: 'Total Units' } } }
			}
		};

		monthlyChart = new Chart(monthlyCanvas, cfg);
	adjustChartYAxis(monthlyChart, values, [], []);
	}

	function renderSkuComparisonChart() {
		if (!forecast || !skuComparisonCanvas) return;

		if (skuComparisonChart) skuComparisonChart.destroy();

		const colors = ['#f97316', '#3b82f6', '#22c55e', '#a855f7', '#eab308', '#ec4899'];
		
		// Get labels from first SKU (downsampled)
		const firstSkuDaily = downsampleForChart(forecast.forecasts[forecast.skus[0]].daily, 300);
		const labels = firstSkuDaily.map((d: any) => d.date);
		
		const datasets = forecast.skus.map((sku: string, idx: number) => {
			const daily = downsampleForChart(forecast.forecasts[sku].daily, 300);
			const values = daily.map((d: any) => d.predicted_units);
			const ma14_client = calculateMovingAverage(values, 14);
			return {
				label: sku,
				data: ma14_client,
				borderColor: colors[idx % colors.length],
				backgroundColor: colors[idx % colors.length] + '20',
				borderWidth: 2,
				pointRadius: 0,
				fill: false
			};
		});

		const cfg: ChartConfiguration = {
			type: 'line',
			data: { labels, datasets },
			options: {
				responsive: true,
				maintainAspectRatio: false,
				animation: false,
				plugins: {
					title: { display: true, text: 'SKU Forecast Comparison', font: { size: 16, weight: 'bold' } },
					legend: { display: true, position: 'top' }
				},
				scales: {
					x: { ticks: { maxTicksLimit: 10 } },
					y: { beginAtZero: true, title: { display: true, text: 'Units' } }
				}
			}
		};

		skuComparisonChart = new Chart(skuComparisonCanvas, cfg);
		try {
			if (skuComparisonChart && skuComparisonChart.data && skuComparisonChart.data.datasets) {
				skuComparisonChart.data.datasets.forEach((d: any) => { d.pointRadius = 0; d.tension = 0.15; d.cubicInterpolationMode = 'monotone'; });
				skuComparisonChart.update();
				// Adjust Y to ensure differences are visible
				try {
					const allValues = skuComparisonChart.data.datasets.flatMap((ds:any)=>ds.data as number[]);
					adjustChartYAxis(skuComparisonChart, allValues, [], []);
				} catch (err) { console.warn('SKU y-axis adjust failed', err); }
			}
		} catch (e) { console.warn('SKU comparison post-adjust failed', e); }
	}

	function renderDistributionChart() {
		if (!forecast || !distributionCanvas || selectedSku === 'all') return;

		if (distributionChart) distributionChart.destroy();

		const dailyData = forecast.forecasts[selectedSku].daily;
		const values = dailyData.map((d: any) => d.predicted_units);

		const min = Math.min(...values);
		const max = Math.max(...values);
		const binCount = 15;
		const binSize = (max - min) / binCount || 1;
		const bins = Array(binCount).fill(0);
		const binLabels = [];

		for (let i = 0; i < binCount; i++) {
			const binStart = min + i * binSize;
			const binEnd = binStart + binSize;
			binLabels.push(`${binStart.toFixed(0)}-${binEnd.toFixed(0)}`);
		}

		values.forEach((v: number) => {
			const binIdx = Math.min(Math.floor((v - min) / binSize), binCount - 1);
			if (binIdx >= 0 && binIdx < binCount) bins[binIdx]++;
		});

		const cfg: ChartConfiguration = {
			type: 'bar',
			data: {
				labels: binLabels,
				datasets: [{
					label: 'Frequency',
					data: bins,
					backgroundColor: 'rgba(168, 85, 247, 0.7)',
					borderColor: '#a855f7',
					borderWidth: 2
				}]
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				animation: false,
				plugins: {
					title: { display: true, text: `Forecast Distribution - ${selectedSku}`, font: { size: 16, weight: 'bold' } },
					legend: { display: false }
				},
				scales: {
					x: { title: { display: true, text: 'Units Range' }, ticks: { maxRotation: 45, minRotation: 45 } },
					y: { beginAtZero: true, title: { display: true, text: 'Days Count' } }
				}
			}
		};

		distributionChart = new Chart(distributionCanvas, cfg);
	}

	$: if (forecast && chartCanvas) {
		console.log('Reactive: calling renderChart');
		setTimeout(() => renderChart(), 0);
	}

	$: if (forecast && quarterlyChartCanvas && selectedSku !== 'all') {
		console.log('Reactive: calling renderQuarterlyChart');
		setTimeout(() => renderQuarterlyChart(), 0);
	}

	$: if (forecast && weekdayCanvas && selectedSku !== 'all') {
		setTimeout(() => renderWeekdayChart(), 0);
	}

	$: if (forecast && monthlyCanvas && selectedSku !== 'all') {
		setTimeout(() => renderMonthlyChart(), 0);
	}

	$: if (forecast && skuComparisonCanvas) {
		setTimeout(() => renderSkuComparisonChart(), 0);
	}

	$: if (forecast && distributionCanvas && selectedSku !== 'all') {
		setTimeout(() => renderDistributionChart(), 0);
	}
</script>

<svelte:head>
	<title>Future Demand Forecast - E-commerce Demand Forecasting</title>
</svelte:head>

<div class="container">
	<h1>Future Demand Forecast</h1>
	<p class="subtitle">Predict upcoming demand for the upcoming financial quarters per product</p>

	<div class="info-box">
		<strong>Note:</strong> Make sure you've trained a model first in the
		<a href="/training">Training</a> section. This will forecast FUTURE demand (not historical).
	</div>

	<div class="upload-section">
		<div class="file-input-group">
			<label for="data-file">
				<span class="label-text">Data Upload</span>
				<input
					id="data-file"
					type="file"
					accept=".csv"
					on:change={handleFileChange}
					disabled={loading}
				/>
				{#if dataFile}
					<span class="file-name">✓ {dataFile.name}</span>
				{/if}
			</label>
		</div>

		<div class="options-row">
			<label class="option-item">
				<span class="option-label">Number of Quarters:</span>
				<input type="number" min="1" max="8" bind:value={quarters} disabled={loading} />
			</label>
		</div>
	</div>

	<button class="predict-btn" on:click={handlePredict} disabled={loading || !dataFile}>
		{#if loading}
			<span class="spinner"></span>
			Generating Future Forecast...
		{:else}
			Generate Future Forecast
		{/if}
	</button>

	{#if error}
		<div class="error-box">
			<strong>Error:</strong> {error}
		</div>
	{/if}

	{#if forecast}
		<div class="results">
			<h2>Forecast Results</h2>

			<div class="sku-selector">
				<label>
					<span>View SKU:</span>
					<select bind:value={selectedSku}>
						<option value="all">All Products (Total)</option>
						{#each forecast.skus as sku}
							<option value={sku}>{sku}</option>
						{/each}
					</select>
				</label>
			</div>

			{#if selectedSku !== 'all'}
				<div class="quarterly-summary">
					<h3>Quarterly Demand Summary - {selectedSku}</h3>
					<div class="quarters-grid">
						{#each forecast.forecasts[selectedSku].quarterly as quarter}
							<div class="quarter-card">
								<div class="quarter-label">{quarter.quarter_label}</div>
								<div class="quarter-value">{quarter.predicted_units.toFixed(0)} units</div>
								<div class="quarter-dates">
									{new Date(quarter.start_date).toLocaleDateString()} - {new Date(
										quarter.end_date
									).toLocaleDateString()}
								</div>
							</div>
						{/each}
					</div>
				</div>

				<div class="chart-container">
					<canvas bind:this={quarterlyChartCanvas}></canvas>
				</div>
			{/if}

			<div class="chart-container">
				<canvas bind:this={chartCanvas}></canvas>
			</div>

			{#if selectedExplanation}
				<ExplanationCard explanation={selectedExplanation} />
			{/if}

			<div class="chart-container">
				<canvas bind:this={skuComparisonCanvas}></canvas>
			</div>

			{#if selectedSku !== 'all'}
				<div class="charts-grid">
					<div class="chart-container-small">
						<canvas bind:this={weekdayCanvas}></canvas>
					</div>
					<div class="chart-container-small">
						<canvas bind:this={monthlyCanvas}></canvas>
					</div>
				</div>
				<div class="chart-container">
					<canvas bind:this={distributionCanvas}></canvas>
				</div>
			{/if}

			<div class="stats">
				<div class="stat-card">
					<span class="stat-label">Model Used</span>
					<span class="stat-value" style="font-size: 1.4rem">{forecast.model}</span>
				</div>
				<div class="stat-card">
					<span class="stat-label">Forecast Horizon</span>
					<span class="stat-value" style="font-size: 1.2rem">{forecast.forecast_horizon}</span>
				</div>
				<div class="stat-card">
					<span class="stat-label">Total SKUs</span>
					<span class="stat-value">{forecast.skus.length}</span>
				</div>
				<div class="stat-card">
					<span class="stat-label">Avg Daily Demand</span>
					<span class="stat-value"
						>{(
							(selectedSku === 'all'
								? forecast.total_daily
								: forecast.forecasts[selectedSku].daily
							).reduce((sum: number, d: any) => sum + d.predicted_units, 0) /
							(selectedSku === 'all'
								? forecast.total_daily.length
								: forecast.forecasts[selectedSku].daily.length)
						).toFixed(1)}</span
					>
				</div>
			</div>

			<button class="download-btn" on:click={downloadPredictions}>
				Download Forecast CSV ({selectedSku})
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
		margin-bottom: 1.5rem;
	}

	.info-box {
		background: #e0f2fe;
		border: 1px solid #0ea5e9;
		padding: 1rem;
		border-radius: 8px;
		margin-bottom: 2rem;
		color: #0c4a6e;
	}

	.info-box a {
		color: #0284c7;
		font-weight: 600;
		text-decoration: underline;
	}

	.upload-section {
		margin-bottom: 2rem;
	}

	.file-input-group {
		border: 2px dashed #ddd;
		border-radius: 8px;
		padding: 1.5rem;
		transition: border-color 0.3s;
	}

	.file-input-group:hover {
		border-color: #f97316;
	}

	.label-text {
		display: block;
		font-weight: 600;
		margin-bottom: 0.5rem;
		color: #333;
		font-size: 0.95rem;
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

	.predict-btn {
		width: 100%;
		padding: 1rem;
		background: #f97316;
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

	.predict-btn:hover:not(:disabled) {
		background: #ea580c;
	}

	.predict-btn:disabled {
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
		margin-bottom: 1rem;
		color: #1a1a1a;
	}

	.stats {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 1rem;
		margin-bottom: 2rem;
	}

	.stat-card {
		background: white;
		padding: 1.2rem;
		border-radius: 8px;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
		text-align: center;
	}

	.stat-label {
		display: block;
		font-size: 0.85rem;
		color: #666;
		margin-bottom: 0.5rem;
	}

	.stat-value {
		display: block;
		font-size: 1.8rem;
		font-weight: 700;
		color: #f97316;
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
		height: 450px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
	}

	.chart-container canvas {
		max-height: 100%;
	}

	.charts-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
		gap: 1rem;
		margin-bottom: 1.5rem;
	}

	.chart-container-small {
		background: white;
		padding: 1.5rem;
		border-radius: 8px;
		height: 350px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
	}

	.chart-container-small canvas {
		max-height: 100%;
	}

	.options-row {
		background: #f8f9fa;
		padding: 1rem;
		border-radius: 8px;
		margin-top: 1rem;
		display: flex;
		gap: 2rem;
		flex-wrap: wrap;
	}

	.option-item {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.option-label {
		font-weight: 600;
		color: #333;
	}

	.option-item input[type='number'],


	.sku-selector {
		background: #e0f2fe;
		padding: 1rem;
		border-radius: 8px;
		margin-bottom: 1.5rem;
	}

	.sku-selector label {
		display: flex;
		align-items: center;
		gap: 1rem;
		font-weight: 600;
		color: #0c4a6e;
	}

	.sku-selector select {
		padding: 0.5rem 1rem;
		border: 2px solid #0ea5e9;
		border-radius: 6px;
		font-size: 1rem;
		background: white;
		cursor: pointer;
	}

	.quarterly-summary {
		background: white;
		padding: 1.5rem;
		border-radius: 8px;
		margin-bottom: 1.5rem;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
	}

	.quarterly-summary h3 {
		margin-bottom: 1.5rem;
		color: #1a1a1a;
		font-size: 1.2rem;
	}

	.quarters-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 1rem;
	}

	.quarter-card {
		background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
		color: white;
		padding: 1.5rem;
		border-radius: 8px;
		text-align: center;
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
	}

	.quarter-label {
		font-size: 0.9rem;
		font-weight: 600;
		opacity: 0.9;
		margin-bottom: 0.5rem;
	}

	.quarter-value {
		font-size: 2rem;
		font-weight: 700;
		margin-bottom: 0.5rem;
	}

	.quarter-dates {
		font-size: 0.75rem;
		opacity: 0.85;
	}
</style>
