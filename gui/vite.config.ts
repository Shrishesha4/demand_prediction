import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const allowedHostsEnv = process.env.ALLOWED_HOSTS || 'pdp.shrishesha.space,pdpl.shrishesha.space,localhost';
const parseAllowedHosts = (v: string) => {
	if (!v) return ['localhost'];
	if (v === 'all' || v === '*') return 'all' as any;
	return v.split(',').map(s => s.trim()).filter(Boolean);
};
const allowedHosts = parseAllowedHosts(allowedHostsEnv);

const hostBinding: boolean | string = process.env.HOST ? process.env.HOST : true;

// Public host/protocol inference for HMR clients (set via FRONTEND_URL or PUBLIC_HOST)
const rawFrontendUrl = process.env.FRONTEND_URL || '';
let publicHost = process.env.PUBLIC_HOST || process.env.HMR_HOST;
let hmrProtocol = process.env.HMR_PROTOCOL || (process.env.HTTPS ? 'wss' : 'ws');
let defaultClientPort = (hmrProtocol === 'wss' ? 443 : 5173);

// If FRONTEND_URL is set, prefer it and infer protocol and port
if (rawFrontendUrl) {
	try {
		const parsed = new URL(rawFrontendUrl);
		if (parsed.hostname) publicHost = parsed.hostname;
		if (parsed.protocol === 'https:') {
			hmrProtocol = process.env.HMR_PROTOCOL || 'wss';
			defaultClientPort = process.env.HMR_CLIENT_PORT ? Number(process.env.HMR_CLIENT_PORT) : 443;
		} else if (parsed.protocol === 'http:') {
			hmrProtocol = process.env.HMR_PROTOCOL || 'ws';
			defaultClientPort = process.env.HMR_CLIENT_PORT ? Number(process.env.HMR_CLIENT_PORT) : 5173;
		}
	} catch (e) {
		// ignore invalid URL and fall back to provided envs
	}
}

// Bind host for the server (where Vite listens) - leave undefined to use server.host
const hmrBindHost = process.env.HMR_BIND_HOST || undefined;
const hmrServerPort = process.env.HMR_PORT ? Number(process.env.HMR_PORT) : 5173;
const hmrClientPort = process.env.HMR_CLIENT_PORT ? Number(process.env.HMR_CLIENT_PORT) : defaultClientPort;
// Client-facing host (what the browser should connect to). Do not allow 0.0.0.0 here.
let hmrClientHost = process.env.HMR_HOST || publicHost;
if (hmrClientHost === '0.0.0.0' || hmrClientHost === '::' || hmrClientHost === '') {
	hmrClientHost = publicHost;
}

// Debug log during config build (appears when starting Vite)
console.log('VITE HMR config:', { hmrProtocol, hmrClientHost, hmrClientPort, hmrServerPort, hmrBindHost });
export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		host: hostBinding,
		allowedHosts: allowedHosts as any,
		hmr: {
			protocol: hmrProtocol as any,
			host: hmrClientHost,
			port: hmrServerPort,
			clientPort: hmrClientPort,
		},
	},
	preview: {
		host: hostBinding,
		allowedHosts: allowedHosts as any,
	}
});
