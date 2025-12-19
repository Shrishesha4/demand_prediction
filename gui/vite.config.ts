import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// Read allowed hosts from environment (comma-separated list) so we can configure the domain via compose
// Example: ALLOWED_HOSTS="pdp.s4home.dpdns.org,localhost"
const allowedHostsEnv = process.env.ALLOWED_HOSTS || 'pdp.shrishesha.space,localhost,0.0.0.0,::1,127.0.0.1';
const parseAllowedHosts = (v: string) => {
	if (!v) return ['localhost'];
	if (v === 'all' || v === '*') return 'all' as any;
	return v.split(',').map(s => s.trim()).filter(Boolean);
};
const allowedHosts = parseAllowedHosts(allowedHostsEnv);

// Allow the HOST env var to control interface binding; default to listening on all interfaces
const hostBinding: boolean | string = process.env.HOST ? process.env.HOST : true;

// Public host used by HMR clients (set to your proxied domain when behind reverse proxy)
const publicHost = process.env.PUBLIC_HOST || 'pdp.shrishesha.space';
const hmrProtocol = process.env.HMR_PROTOCOL || (process.env.HTTPS ? 'wss' : 'ws');

// Distinguish HMR server bind vs client connection:
// - HMR server should bind to an address available on this machine (default: 0.0.0.0)
// - HMR client should connect to the public host/port seen by the browser (default port 443 for wss)
const hmrServerHost = process.env.HMR_BIND_HOST || undefined; // e.g. 0.0.0.0 (or leave undefined to use server.host)
const hmrServerPort = process.env.HMR_PORT ? Number(process.env.HMR_PORT) : 5173;
const hmrClientPort = process.env.HMR_CLIENT_PORT
	? Number(process.env.HMR_CLIENT_PORT)
	: (hmrProtocol === 'wss' ? 443 : 5173);

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		host: hostBinding,
		allowedHosts: allowedHosts as any,
		hmr: Object.assign(
			{
				protocol: hmrProtocol as any,
				port: hmrServerPort,
				clientPort: hmrClientPort,
			},
			// Only set a bind host for the HMR server if explicitly requested so
			// we don't attempt to bind to the public DNS/IP (which causes EADDRNOTAVAIL).
			(hmrServerHost ? { host: hmrServerHost } : {})
		),
	},
	preview: {
		host: hostBinding,
		allowedHosts: allowedHosts as any,
	}
});
