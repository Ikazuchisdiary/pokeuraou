/**
 * The Showdown commit a dump was built from, or a stop.
 *
 * The dumps record `meta.showdownCommit` so a reader can tell which dex produced them.
 * The old reader ran `git -C vendor/pokemon-showdown rev-parse HEAD` and trusted the answer,
 * but when the submodule is not initialised (a fresh worktree) that directory is not a
 * repository of its own, git walks up to the parent, and the parent's HEAD was written into
 * the dump as if it were Showdown's (IKA-152). A failure was also written, as 'unknown'.
 *
 * Now the commit is read only from a directory that is itself the top of a git checkout, and
 * it must equal the gitlink the parent records for the submodule (the index entry, so a
 * staged bump can be dumped before it is committed). Anything else throws: the dump is not
 * written with a commit nobody can vouch for.
 */
import * as fs from 'fs';
import * as path from 'path';
import { execFileSync } from 'child_process';

export const SHOWDOWN_SUBMODULE = 'vendor/pokemon-showdown';

function git(cwd: string, args: string[]): string {
	return execFileSync('git', ['-C', cwd, ...args], {
		encoding: 'utf8',
		stdio: ['ignore', 'pipe', 'pipe'],
	}).trim();
}

function samePath(a: string, b: string): boolean {
	const norm = (p: string) => {
		const real = fs.realpathSync.native(path.resolve(p));
		return process.platform === 'win32' ? real.toLowerCase() : real;
	};
	return norm(a) === norm(b);
}

/** The commit the parent repository's index records for the Showdown submodule. */
export function showdownGitlink(root: string): string {
	let line: string;
	try {
		line = git(root, ['ls-files', '--stage', '--', SHOWDOWN_SUBMODULE]);
	} catch (e) {
		throw new Error(`cannot read the gitlink of ${SHOWDOWN_SUBMODULE} in ${root}: ${String(e)}`);
	}
	// "160000 <sha> 0\tvendor/pokemon-showdown"
	const m = /^160000 ([0-9a-f]{40}) 0\t/.exec(line);
	if (!m) {
		throw new Error(`${SHOWDOWN_SUBMODULE} is not a submodule gitlink in ${root} (ls-files: ${JSON.stringify(line)})`);
	}
	return m[1];
}

/**
 * HEAD of the Showdown checkout at `showdownDir`, checked against the parent's gitlink.
 *
 * `showdownDir` is the directory the dex is actually loaded from; it must be the top of its
 * own git checkout, or git would answer for whatever repository contains it.
 */
export function showdownCommit(root: string, showdownDir: string): string {
	const expected = showdownGitlink(root);
	if (!fs.existsSync(showdownDir)) {
		throw new Error(`Showdown directory ${showdownDir} does not exist; run git submodule update --init`);
	}
	let top: string;
	try {
		top = git(showdownDir, ['rev-parse', '--show-toplevel']);
	} catch (e) {
		throw new Error(`${showdownDir} is not inside a git checkout: ${String(e)}`);
	}
	if (!samePath(top, showdownDir)) {
		throw new Error(
			`${showdownDir} is not a git checkout of its own (git answers for ${top}); ` +
			'the submodule is not initialised -- run git submodule update --init'
		);
	}
	const head = git(showdownDir, ['rev-parse', 'HEAD']);
	if (head !== expected) {
		throw new Error(
			`Showdown at ${showdownDir} is at ${head}, but ${SHOWDOWN_SUBMODULE}'s gitlink is ${expected}; ` +
			'check out the gitlink (git submodule update) or stage the bump (git add) before dumping'
		);
	}
	return head;
}
