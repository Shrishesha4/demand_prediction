<script lang="ts">
	import { goto } from '$app/navigation';
</script>

<svelte:head>
	<title>E-commerce Demand Forecasting</title>
</svelte:head>

<div class="container">
	<header class="hero">
		<h1>E-commerce Demand Forecasting</h1>
		<p class="tagline">Product demand prediction using ARIMA and LSTM models</p>
	</header>

	<div class="cards">
		<div class="card">
			<h2>Train</h2>
			<p>Upload your training and test datasets to train ARIMA and LSTM forecasting models.</p>
			<ul class="features">
				<li>✓ ARIMA with weekly seasonality</li>
				<li>✓ Multi-step LSTM with time features</li>
				<li>✓ Performance metrics and comparison</li>
				<li>✓ Model persistence for reuse</li>
			</ul>
			<button class="cta-btn" on:click={() => goto('/training')}>Train →</button>
		</div>

		<div class="card">
			<h2>Predict</h2>
			<p>Use your trained LSTM model to generate demand forecasts on new data.</p>
			<ul class="features">
				<li>✓ Upload any e-commerce dataset</li>
				<li>✓ Instant predictions</li>
				<li>✓ Downloadable forecast CSV</li>
				<li>✓ Visual insights and statistics</li>
			</ul>
			<button class="cta-btn" on:click={() => goto('/predict')}>Predict →</button>
		</div>
	</div>

	<div class="info-section">
		<h4>How It Works</h4>
		<div class="steps">
			<div class="step">
				<div class="step-num">1</div>
				<h4>Prepare Data</h4>
				<p>
					Your CSV must have columns: <code>date</code>, <code>sku_id</code>,
					<code>units_sold</code>, <code>price</code>, <code>promotion_flag</code>
				</p>
			</div>
			<div class="step">
				<div class="step-num">2</div>
				<h4>Train</h4>
				<p>
					Upload train and test CSVs in the Training section. The system trains both ARIMA and
					LSTM models and compares their performance.
				</p>
			</div>
			<div class="step">
				<div class="step-num">3</div>
				<h4>Predict</h4>
				<p>
					Upload new data in the Prediction section to generate forecasts using the trained LSTM
					model. Download results as CSV.
				</p>
			</div>
		</div>
	</div>

	<!-- Project & Contributors -->
	<section class="project-section">
		<h3>Demand Forecasting for E-Commerce Using Python LSTM Models Compared with ARIMA to Enhance Inventory Optimization</h3>
		<p class="project-desc">Leveraging classical time-series modeling and deep learning to forecast e-commerce demand by benchmarking ARIMA against LSTM, with the explicit intent of identifying accuracy trade-offs under linear versus nonlinear demand dynamics. Implement a Python-based forecasting pipeline that ingests historical sales data, performs statistical stationarity and sequence modeling, and quantitatively compares ARIMA and LSTM outputs to drive data-backed inventory optimization decisions.</p>

		<h4>Contributors & Module assignments</h4>
		<div class="contributors">
			<div class="contributor">
				<div class="contrib-meta">
					<span class="contrib-name">Shrishesha Narmatesshvara</span>
					<span class="contrib-id">192321183</span>
				</div>
				<div class="contrib-module">Module 3 — LSTM Model</div>
				<p class="contrib-desc">Responsible for LSTM architecture, training strategy, model persistence, and integration with the API for production forecasting and per-SKU distribution logic.</p>
			</div>

			<div class="contributor">
				<div class="contrib-meta">
					<span class="contrib-name">Anumaneni Somnath</span>
					<span class="contrib-id">192472386</span>
				</div>
				<div class="contrib-module">Module 2 — ARIMA Forecasting</div>
				<p class="contrib-desc">Focuses on SARIMAX modeling, seasonal and stationarity analysis, model selection, and benchmarking classical methods against LSTM.</p>
			</div>

			<div class="contributor">
				<div class="contrib-meta">
					<span class="contrib-name">K. Pavan Kumar</span>
					<span class="contrib-id">192212273</span>
				</div>
				<div class="contrib-module">Module 1 — Data Preprocessing &amp; Feature Engineering</div>
				<p class="contrib-desc">Leads data ingestion, cleaning, aggregation and feature engineering (including cyclical time features and handling missing values) that enable both ARIMA and LSTM experiments.</p>
			</div>
		</div>
	</section>

	<section class="course-info">
		<p class="course-name">Python Programming for Expert Systems</p>
		<p class="course-code">Course Code: <strong>CSA0804</strong></p>
	</section>

	<footer>
		<p>
			Built with <strong>FastAPI</strong> + <strong>Svelte</strong> + <strong>TensorFlow</strong> +
			<strong>Statsmodels</strong>
		</p>
	</footer>
</div>

<style>
	/* Minimalistic / Scientific theme: neutral surfaces, subtle accents, high legibility */
	:global(body) {
		margin: 0;
		font-family: Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue',
			Arial;
		background: #f5f7fa; /* soft neutral */
		color: #0f172a; /* dark slate for text */
		-webkit-font-smoothing: antialiased;
		-moz-osx-font-smoothing: grayscale;
		min-height: 100vh;
	}

	.container {
		max-width: 1100px;
		margin: 0 auto;
		padding: 2.25rem 1.25rem;
	}

	.hero {
		text-align: left;
		margin-bottom: 2.5rem;
	}

	.hero h1 {
		font-size: 2.25rem;
		margin: 0 0 0.25rem 0;
		color: #0b1220;
		font-weight: 700;
		letter-spacing: -0.01em;
	}

	.tagline {
		font-size: 1rem;
		color: #334155;
		margin-bottom: 0.5rem;
	}

	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
		gap: 1rem;
		margin-bottom: 2rem;
	}

	.card {
		background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
		padding: 1.25rem;
		border-radius: 10px;
		border: 1px solid #e6eef6;
		transition: transform 0.16s ease, box-shadow 0.16s ease;
	}

	.card:hover {
		transform: translateY(-4px);
		box-shadow: 0 6px 18px rgba(12, 20, 28, 0.06);
	}

	.card h2 {
		font-size: 1.1rem;
		margin: 0.25rem 0 0.5rem 0;
		color: #0b1220;
		font-weight: 700;
	}

	.card p {
		color: #475569;
		margin: 0 0 1rem 0;
		line-height: 1.45;
		font-size: 0.95rem;
	}

	.features {
		list-style: none;
		padding: 0;
		margin: 0 0 1rem 0;
		color: #445569;
		font-size: 0.92rem;
	}

	.info-section h4 {
		font-size: 0.98rem;
		margin: 0.5rem 0 0.75rem 0;
		color: #0b1220;
		font-weight: 700;
		letter-spacing: -0.01em;
		background: linear-gradient(90deg, rgba(11,95,255,0.06), rgba(11,95,255,0.02));
		padding: 0.25rem 0.5rem;
		border-radius: 6px;
		display: inline-block;
	}


	.features li { padding: 0.25rem 0; }

	/* Clear, visible CTA button with good contrast */
	.cta-btn {
		width: 100%;
		padding: 0.9rem 1rem;
		background: #0b5fff; /* crisp scientific blue */
		color: #ffffff;
		border: 1px solid rgba(11, 95, 255, 0.12);
		border-radius: 8px;
		font-size: 1rem;
		font-weight: 700;
		letter-spacing: 0.01em;
		cursor: pointer;
		box-shadow: 0 6px 18px rgba(11, 95, 255, 0.06);
		transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
	}

	.cta-btn:hover {
		transform: translateY(-2px);
		background: #054be6;
		box-shadow: 0 10px 24px rgba(11, 95, 255, 0.10);
	}

	/* Info section and steps are cleaner with subtle border */
	.info-section {
		background: transparent;
		padding: 1.25rem;
		border-radius: 8px;
		margin-bottom: 1.75rem;
		border-top: 1px solid rgba(15, 23, 42, 0.04);
	}

	.info-section h3 {
		text-align: left;
		font-size: 1.05rem;
		margin-bottom: 1rem;
		color: #0b1220;
	}

	.steps {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 0.75rem;
	}

	.step { text-align: left; }

	.step-num {
		width: 40px;
		height: 40px;
		background: #eef6ff;
		color: #0b5fff;
		border-radius: 50%;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		font-size: 1rem;
		font-weight: 700;
		margin-right: 0.75rem;
	}

	.step h4 { font-size: 1rem; margin: 0 0 0.25rem 0; color: #0b1220; }
	.step p { color: #475569; margin: 0; font-size: 0.92rem; }

	code {
		background: #f8fafc;
		padding: 0.18rem 0.36rem;
		border-radius: 4px;
		font-size: 0.9em;
		color: #0b1220;
	}

	footer { text-align: center; color: #334155; padding: 1.5rem 0; font-size: 0.92rem; }

	@media (max-width: 768px) {
		.hero h1 { font-size: 1.5rem; }
		.cards { grid-template-columns: 1fr; gap: 0.75rem; }
		.container { padding: 1rem; }
	}

	/* Project & contributors styling (improved spacing & highlights) */
	.project-section {
		background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
		padding: 1.5rem;
		border-radius: 12px;
		border: 1px solid #e6eef6;
		margin-bottom: 1.75rem;
	}
	/* Section heading pill */
	.project-section h3 {
		display: inline-block;
		background: #eef6ff;
		color: #0b5fff;
		padding: 0.35rem 0.75rem;
		border-radius: 8px;
		margin: 0 0 0.6rem 0;
		font-size: 1.08rem;
		font-weight: 700;
	}
	/* Slightly enhanced h4 for subsection headings */
	.project-section h4 {
		font-size: 0.98rem;
		margin: 0.5rem 0 0.75rem 0;
		color: #0b1220;
		font-weight: 700;
		letter-spacing: -0.01em;
		background: linear-gradient(90deg, rgba(11,95,255,0.06), rgba(11,95,255,0.02));
		padding: 0.25rem 0.5rem;
		border-radius: 6px;
		display: inline-block;
	}

	.project-desc {
		color: #475569;
		margin-bottom: 1.25rem;
		line-height: 1.6;
	}
	.contributors {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
		gap: 1rem;
		align-items: start;
	}
	.contributor {
		background: #ffffff;
		padding: 1.25rem;
		border-radius: 10px;
		border: 1px solid #edf2f7;
		box-shadow: 0 6px 18px rgba(11, 95, 255, 0.04);
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.contrib-meta {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.5rem;
		margin-bottom: 0.35rem;
	}
	.contrib-name {
		font-weight: 800;
		color: #0b1220;
		font-size: 1rem;
	}
	.contrib-id {
		background: #f8fafc;
		padding: 0.18rem 0.45rem;
		border-radius: 6px;
		color: #64748b;
		font-size: 0.9rem;
	}
	.contrib-module {
		font-weight: 700;
		color: #0b1220;
		margin-top: 0.25rem;
		padding: 0.28rem 0.5rem;
		background: #f1f9ff;
		border-radius: 6px;
		display: inline-block;
	}
	.contrib-desc {
		color: #475569;
		margin: 0;
		font-size: 0.95rem;
		line-height: 1.4;
	}

	/* Course info */
	.course-info { text-align: center; margin-top: 1rem; margin-bottom: 0.5rem; }
	.course-name { font-weight: 700; color: #0b1220; margin: 0; }
	.course-code { color: #475569; margin: 0; }
</style>
