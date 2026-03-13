const DECOR_ITEMS = Array.from({ length: 10 }, (_, index) => index);

export default function FestiveOverlay({ festival }) {
  return (
    <div className={`festive-overlay ${festival ? `theme-${festival.effect}` : 'theme-default'}`}>
      <div className="scene-orb orb-a" />
      <div className="scene-orb orb-b" />
      <div className="scene-orb orb-c" />
      <div className="scene-grid" />
      {festival?.effect === 'christmas' && (
        <div className="snow-field">
          {DECOR_ITEMS.map((item) => (
            <span key={`snow-${item}`} className={`snowflake snowflake-${item}`} />
          ))}
        </div>
      )}
      {festival?.effect === 'diwali' && (
        <div className="spark-field">
          {DECOR_ITEMS.map((item) => (
            <span key={`spark-${item}`} className={`spark spark-${item}`} />
          ))}
        </div>
      )}
      {festival?.effect === 'ramadan' && (
        <div className="moon-scene">
          <div className="crescent" />
          <div className="lantern lantern-left" />
          <div className="lantern lantern-right" />
        </div>
      )}
      {festival?.effect === 'holi' && (
        <div className="color-bursts">
          {DECOR_ITEMS.slice(0, 6).map((item) => (
            <span key={`burst-${item}`} className={`burst burst-${item}`} />
          ))}
        </div>
      )}
      {(festival?.effect === 'navratri' || festival?.effect === 'durgapuja') && (
        <div className="festival-ring">
          <div className="ring ring-outer" />
          <div className="ring ring-inner" />
        </div>
      )}
    </div>
  );
}
