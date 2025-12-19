import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const allowedHostsEnv = process.env.ALLOWED_HOSTS || 'pdp.shrishesha.space,localhost,0.0.0.0,::1,127.0.0.1';
const parseAllowedHosts = (v: string) => {
	if (!v) return ['localhost'];
	if (v === 'all' || v === '*') return 'all' as any;
	return v.split(',').map(s => s.trim()).filter(Boolean);
};
const allowedHosts = parseAllowedHosts(allowedHostsEnv);

const hostBinding: boolean | string = process.env.HOST ? process.env.HOST : true;

const hmrProtocol = process.env.HMR_PROTOCOL || (process.env.HTTPS ? 'wss' : 'ws');

const hmrServerHost = process.env.HMR_BIND_HOST || undefined;
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
			(hmrServerHost ? { host: hmrServerHost } : {})
		),
	},
	preview: {
		host: hostBinding,
		allowedHosts: allowedHosts as any,
	}
});
