import { useState, useMemo } from 'react';
import type { Player } from '@the-duke/engine';
import { Board } from './components/Board.js';
import { GameControls } from './components/GameControls.js';
import { BagPanel } from './components/BagPanel.js';
import { TileDetail } from './components/TileDetail.js';
import { useGame } from './hooks/useGame.js';
import { useIsMobile } from './hooks/useIsMobile.js';

export type GameMode = 'hotseat' | 'vs-random' | 'vs-minimax';

export function App() {
  const [mode, setMode] = useState<GameMode>('vs-minimax');
  const [aiPlayer, setAiPlayer] = useState<Player>('P2');
  const game = useGame(mode, aiPlayer);
  const isMobile = useIsMobile(660);

  const { state } = game;
  const isSetup = state.status === 'setup';

  const commandTargetTile = useMemo(() => {
    if (!game.commandTarget) return null;
    const id = state.board[game.commandTarget.row][game.commandTarget.col];
    return id ? state.tiles.get(id) ?? null : null;
  }, [game.commandTarget, state]);

  const boardEl = (
    <Board
      state={state}
      selectedTile={game.selectedTile}
      legalMoves={game.legalMoves}
      onCellClick={game.onCellClick}
      drawMode={game.drawMode}
      placementTargets={game.placementTargets}
      commandTarget={game.commandTarget}
      commandDestinations={game.commandDestinations}
      setupTargets={game.setupTargets}
      inCheck={game.inCheck}
      viewingEnemyMoves={game.viewingEnemyMoves}
      allFriendlyTargets={game.allFriendlyTargets}
      allEnemyTargets={game.allEnemyTargets}
    />
  );

  const p1Bag = (
    <BagPanel
      player="P1"
      bag={state.bags.P1}
      isActive={state.currentPlayer === 'P1'}
      viewingTile={game.viewingBagTile?.owner === 'P1' ? game.viewingBagTile.name : null}
      onTileClick={game.onBagTileClick}
      compact={isMobile}
    />
  );

  const p2Bag = (
    <BagPanel
      player="P2"
      bag={state.bags.P2}
      isActive={state.currentPlayer === 'P2'}
      viewingTile={game.viewingBagTile?.owner === 'P2' ? game.viewingBagTile.name : null}
      onTileClick={game.onBagTileClick}
      compact={isMobile}
    />
  );

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: isMobile ? '0.4rem' : '0.6rem',
      width: '100%',
      maxWidth: '820px',
      height: '100%',
      overflow: 'hidden',
    }}>
      {/* Title */}
      <h1 style={{
        fontFamily: 'var(--font-display)',
        fontSize: isMobile ? '1.3rem' : '1.7rem',
        fontWeight: 700,
        letterSpacing: '0.12em',
        color: 'var(--accent)',
        textAlign: 'center',
      }}>
        THE DUKE
      </h1>

      <GameControls
        mode={mode}
        onModeChange={setMode}
        aiPlayer={aiPlayer}
        onAiPlayerChange={setAiPlayer}
        state={state}
        onNewGame={game.newGame}
        onUndo={game.undo}
        canUndo={game.canUndo}
        compact={isMobile}
        showAllMoves={game.showAllMoves}
        onToggleShowAllMoves={game.toggleShowAllMoves}
      />

      {/* Setup instruction banner */}
      {game.setupLabel && (
        <div style={{
          padding: '0.5rem 1.2rem',
          background: 'var(--surface)',
          border: '1.5px solid var(--accent)',
          borderRadius: 'var(--radius)',
          fontSize: isMobile ? '0.75rem' : '0.85rem',
          fontWeight: 600,
          color: 'var(--accent)',
          textAlign: 'center',
          letterSpacing: '0.02em',
          fontFamily: 'var(--font-body)',
        }}>
          {game.setupLabel}
        </div>
      )}

      {/* Desktop: bags flanking board */}
      {!isMobile && (
        <div style={{
          display: 'flex',
          gap: '10px',
          width: '100%',
          alignItems: 'flex-start',
          justifyContent: 'center',
          flex: '1 1 0',
          minHeight: 0,
        }}>
          {!isSetup && p1Bag}
          {boardEl}
          {!isSetup && p2Bag}
        </div>
      )}

      {/* Mobile: board then bags below */}
      {isMobile && (
        <>
          <div style={{
            flex: '1 1 0',
            minHeight: 0,
            width: '100%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'flex-start',
            overflow: 'hidden',
          }}>
            <div style={{
              width: '100%',
              maxWidth: '460px',
              maxHeight: '100%',
              aspectRatio: '1',
            }}>
              {boardEl}
            </div>
          </div>
          {!isSetup && (
            <div style={{
              display: 'flex',
              gap: '6px',
              width: '100%',
              flexShrink: 0,
            }}>
              {p1Bag}
              {p2Bag}
            </div>
          )}
        </>
      )}

      {!isSetup && (
        <TileDetail
          tile={game.selectedTileInstance}
          viewingBagTile={game.viewingBagTile}
          currentPlayer={state.currentPlayer}
          bagSize={state.bags[state.currentPlayer].length}
          canDraw={game.canDraw}
          drawMode={game.drawMode}
          selectedDrawTile={game.selectedDrawTile}
          onStartDraw={game.startDraw}
          commandTarget={game.commandTarget}
          commandTargetTile={commandTargetTile}
          compact={isMobile}
        />
      )}

      {state.status !== 'active' && state.status !== 'setup' && (
        <div style={{
          padding: '0.75rem 2rem',
          background: 'var(--accent)',
          borderRadius: 'var(--radius-lg)',
          fontWeight: 700,
          fontSize: '1.1rem',
          color: '#0d0b14',
          letterSpacing: '0.04em',
          fontFamily: 'var(--font-display)',
        }}>
          {state.status === 'P1_wins' ? 'Light Wins!' : 'Dark Wins!'}
        </div>
      )}
    </div>
  );
}
