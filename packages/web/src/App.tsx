import { useState, useMemo } from 'react';
import { Board } from './components/Board.js';
import { GameControls } from './components/GameControls.js';
import { BagPanel } from './components/BagPanel.js';
import { TileDetail } from './components/TileDetail.js';
import { useGame } from './hooks/useGame.js';

export type GameMode = 'hotseat' | 'vs-random' | 'vs-minimax';

export function App() {
  const [mode, setMode] = useState<GameMode>('hotseat');
  const game = useGame(mode);

  const { state } = game;

  const commandTargetTile = useMemo(() => {
    if (!game.commandTarget) return null;
    const id = state.board[game.commandTarget.row][game.commandTarget.col];
    return id ? state.tiles.get(id) ?? null : null;
  }, [game.commandTarget, state]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '0.75rem',
      width: '100%',
      maxWidth: '720px',
      padding: '0 8px',
    }}>
      <h1 style={{
        fontSize: '1.6rem',
        fontWeight: 700,
        letterSpacing: '0.05em',
        color: 'var(--accent)',
      }}>
        THE DUKE
      </h1>

      <GameControls
        mode={mode}
        onModeChange={setMode}
        state={state}
        onNewGame={game.newGame}
        onUndo={game.undo}
        canUndo={game.canUndo}
      />

      <div style={{
        display: 'flex',
        gap: '8px',
        width: '100%',
        alignItems: 'flex-start',
        justifyContent: 'center',
      }}>
        <BagPanel
          player="P1"
          bag={state.bags.P1}
          isActive={state.currentPlayer === 'P1'}
          viewingTile={game.viewingBagTile?.owner === 'P1' ? game.viewingBagTile.name : null}
          onTileClick={game.onBagTileClick}
        />

        <Board
          state={state}
          selectedTile={game.selectedTile}
          legalMoves={game.legalMoves}
          onCellClick={game.onCellClick}
          drawMode={game.drawMode}
          placementTargets={game.placementTargets}
          commandTarget={game.commandTarget}
          commandDestinations={game.commandDestinations}
        />

        <BagPanel
          player="P2"
          bag={state.bags.P2}
          isActive={state.currentPlayer === 'P2'}
          viewingTile={game.viewingBagTile?.owner === 'P2' ? game.viewingBagTile.name : null}
          onTileClick={game.onBagTileClick}
        />
      </div>

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
      />

      {state.status !== 'active' && (
        <div style={{
          padding: '0.75rem 1.5rem',
          background: 'var(--accent)',
          borderRadius: '8px',
          fontWeight: 600,
          fontSize: '1.1rem',
        }}>
          {state.status === 'P1_wins' ? 'Light Wins!' : 'Dark Wins!'}
        </div>
      )}
    </div>
  );
}
