import { Bot } from "grammy";

const bot = new Bot(process.env.TELEGRAM_BOT_TOKEN);

bot.command("start", (ctx) => ctx.reply("Hi! serquestbot is connected."));
bot.on("message:text", (ctx) => ctx.reply(`You said: ${ctx.message.text}`));

bot.catch((err) => console.error("Bot error:", err));
bot.start();
console.log("Bot is running...");
